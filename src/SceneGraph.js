import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

const SceneGraph = ({ graphData, setGraphData }) => {
  const svgRef = useRef();
  const [tempEdge, setTempEdge] = useState(null); // Temporary edge while dragging
  const draggingNodeRef = useRef(null); // Use a ref to track draggingNode synchronously

  useEffect(() => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      console.warn("Invalid graphData:", graphData);
      return;
    }

    const width = 1000;
    const height = 600;

    // object 노드 필터링
    const objectNodes = graphData.nodes.filter(
      (node) => node.type === "object"
    );

    // 노드 초기 위치 설정
    graphData.nodes.forEach((node, index) => {
      if (node.type === "object") {
        const objectIndex = objectNodes.findIndex(
          (objNode) => objNode.id === node.id
        );
        node.fx = width / 2 - (objectNodes.length - 1) * 75 + objectIndex * 150;
        node.fy = height / 2;
      } else {
        node.fx = null;
        node.fy = null;
      }
    });

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    svg.selectAll("*").remove(); // Clear previous render
    const simulation = d3
      .forceSimulation(graphData.nodes)
      .force(
        "link",
        d3
          .forceLink(graphData.links)
          .id((d) => d.id)
          .distance(150) // 링크 거리 증가
      )
      .force("charge", d3.forceManyBody().strength(-300));
    // Draw edges
    const link = svg
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graphData.links)
      .join("line")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)")
      .on("contextmenu", (event, d) => { // 'd'는 해당 Edge 데이터
        console.log("Delete link", d);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지
    
        // React 상태에서 해당 Edge 삭제
        setGraphData((prevData) => ({
          ...prevData,
          links: prevData.links.filter((link) => link !== d), // 선택된 Edge 삭제
        }));
      });

    // Draw temporary edge (if any)
    if (tempEdge && tempEdge.source && tempEdge.target) {
      svg
        .append("line")
        .attr("class", "temp-line")
        .attr("x1", tempEdge?.source?.x ?? 0) // 값이 없으면 0을 기본값으로 사용
        .attr("y1", tempEdge?.source?.y ?? 0)
        .attr("x2", tempEdge?.target?.x ?? 0)
        .attr("y2", tempEdge?.target?.y ?? 0)

        .attr("stroke", "gray")
        .attr("stroke-dasharray", "4");
    }

    // Draw nodes
    const nodeGroup = svg
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(graphData.nodes)
      .join("g");

    // Add rectangles to nodes
    nodeGroup
      .append("rect")
      .attr("width", (d) => d.name.length * 10 + 20)
      .attr("height", 30)
      .attr("x", (d) => -(d.name.length * 5 + 10))
      .attr("y", -15)
      .attr("fill", (d) =>
        d.type === "object"
          ? "#ff6b6b"
          : d.type === "attribute"
          ? "#4dabf7"
          : "#51cf66"
      )
      .attr("rx", 10)
      .attr("ry", 10)
      .call(
        d3
          .drag()
          .on("start", (event, node) => {
            draggingNodeRef.current = node; // 드래그 시작 시 노드 저장
            setTempEdge({ source: node, target: { x: node.x, y: node.y } }); // tempEdge 초기화
          })
          .on("drag", (event) => {
            if (draggingNodeRef.current) {
              const svg = d3.select(svgRef.current);
              const point = svg.node().createSVGPoint();
              point.x = event.sourceEvent.clientX; // 마우스 스크린 좌표
              point.y = event.sourceEvent.clientY; // 마우스 스크린 좌표

              // 스크린 좌표를 SVG 좌표로 변환
              const transformedPoint = point.matrixTransform(
                svg.node().getScreenCTM().inverse()
              );

              // tempEdge 업데이트
              setTempEdge({
                source: draggingNodeRef.current,
                target: { x: transformedPoint.x, y: transformedPoint.y },
              });
            }
          })
          .on("end", (event) => {
            if (draggingNodeRef.current) {
              // 연결할 노드 찾기
              const svg = d3.select(svgRef.current);
              const point = svg.node().createSVGPoint();
              point.x = event.sourceEvent.clientX; // 스크린 좌표
              point.y = event.sourceEvent.clientY; // 스크린 좌표
          
              // 스크린 좌표를 SVG 로컬 좌표로 변환
              const transformedPoint = point.matrixTransform(
                svg.node().getScreenCTM().inverse()
              );
          
              const targetNode = graphData.nodes.find(
                (n) =>
                  Math.hypot(n.x - transformedPoint.x, n.y - transformedPoint.y) < 30
              );
          
              console.log("end graphData", graphData);
              console.log("end", draggingNodeRef.current, targetNode);
          
              if (targetNode && targetNode !== draggingNodeRef.current) {
                // 새 엣지 추가
                const newEdge = {
                  source: draggingNodeRef.current,
                  target: targetNode,
                };
          
                console.log("newEdge", newEdge);
          
                // React 상태로 링크 업데이트
                setGraphData((prevData) => ({
                  ...prevData,
                  links: [...prevData.links, newEdge],
                }));
              }
          
              // tempEdge 초기화
              setTempEdge(null);
              draggingNodeRef.current = null;
            }
          })          
      )
      .on("contextmenu", (event, node) => {
        console.log("Delete node", node);
        event.preventDefault(); // Prevent default context menu
      
        // React 상태로 노드와 연결된 링크 삭제
        setGraphData((prevData) => ({
          ...prevData,
          nodes: prevData.nodes.filter((n) => n !== node),
          links: prevData.links.filter(
            (link) => link.source !== node && link.target !== node
          ),
        }));
      });
      

    // Add text to nodes
    nodeGroup
      .append("text")
      .attr("text-anchor", "middle")
      .attr("alignment-baseline", "middle")
      .attr("fill", "#fff")
      .style("font-size", "12px")
      .text((d) => d.name);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => {
          const dx = (d.target?.x ?? 0) - (d.source?.x ?? 0);
          const dy = (d.target?.y ?? 0) - (d.source?.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 0 방지
          return (d.source?.x ?? 0) + (dx / distance) * 20; // 20px 띄움
        })
        .attr("y1", (d) => {
          const dx = (d.target?.x ?? 0) - (d.source?.x ?? 0);
          const dy = (d.target?.y ?? 0) - (d.source?.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 0 방지
          return (d.source?.y ?? 0) + (dy / distance) * 20; // 20px 띄움
        })
        .attr("x2", (d) => {
          const dx = (d.target?.x ?? 0) - (d.source?.x ?? 0);
          const dy = (d.target?.y ?? 0) - (d.source?.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 0 방지
          return (d.target?.x ?? 0) - (dx / distance) * 20; // 20px 띄움
        })
        .attr("y2", (d) => {
          const dx = (d.target?.x ?? 0) - (d.source?.x ?? 0);
          const dy = (d.target?.y ?? 0) - (d.source?.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 0 방지
          return (d.target?.y ?? 0) - (dy / distance) * 20; // 20px 띄움
        });

      nodeGroup.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
    });

    // Function to re-render graph
   
  }, [tempEdge, graphData, setGraphData]);

  return (
    <div>
      <svg ref={svgRef}></svg>
      <style>{`
        .link {
          stroke: #aaa;
          stroke-width: 2;
        }
      `}</style>
    </div>
  );
};

export default SceneGraph;
