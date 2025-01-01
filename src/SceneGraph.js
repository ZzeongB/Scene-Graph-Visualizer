import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

const SceneGraph = ({
  graphData,
  setGraphData,
  currentMode,
  inputText,
  setInputText,
}) => {
  const svgRef = useRef();
  const [tempEdge, setTempEdge] = useState(null); // Temporary edge while dragging
  const draggingNodeRef = useRef(null); // Use a ref to track draggingNode synchronously
  const [inputPosition, setInputPosition] = useState(null); // 입력창 위치
  const [inputValue, setInputValue] = useState(""); // 사용자 입력 값
  const [selectedType, setSelectedType] = useState("object"); // 기본 타입 설정

  useEffect(() => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      console.warn("Invalid graphData:", graphData);
      return;
    }

    const width = 1200;
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
          .distance(150)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2)) // 중앙으로 당기는 힘
      .force("collide", d3.forceCollide().radius(70)); // 노드 간 최소 거리 50 설정

    // Draw edges
    const link = svg
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graphData.links)
      .join("line")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");

    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 9)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#aaa");

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
      .attr("ry", 10);

    nodeGroup
      .append("text")
      .attr("text-anchor", "middle")
      .attr("alignment-baseline", "middle")
      .attr("fill", "#fff")
      .style("font-size", "12px")
      .text((d) => d.name);

    // ========================================
    //  Add event listeners based on current mode
    // ========================================
    if (currentMode === "default") {
      nodeGroup.call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; // 고정 위치 설정
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; // 드래그 종료 후 위치 해제
            d.fy = null;
          })
      );
    }

    if (currentMode === "draw") {
      nodeGroup.call(
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
                  Math.hypot(
                    n.x - transformedPoint.x,
                    n.y - transformedPoint.y
                  ) < 30
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
      );
    }

    if (currentMode === "edit") {
      nodeGroup.on("click", (event, node) => {
        console.log("Edit node", node);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        // React 상태에서 해당 노드 수정
        const newName = prompt("Enter new name for the node:", node.name);
        if (newName) {
          setGraphData((prevData) => ({
            ...prevData,
            nodes: prevData.nodes.map((n) =>
              n === node ? { ...n, name: newName } : n
            ),
          }));
        }
      });
    }

    if (currentMode === "delete") {
      nodeGroup.on("click", (event, node) => {
        console.log("Delete node", node);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        // React 상태에서 해당 노드 삭제
        setGraphData((prevData) => ({
          ...prevData,
          nodes: prevData.nodes.filter((n) => n !== node), // 선택된 노드 삭제
          links: prevData.links.filter(
            (link) => link.source !== node && link.target !== node
          ), // 선택된 노드와 연결된 링크 삭제
        }));
      });

      link.on("click", (event, d) => {
        // 'd'는 해당 Edge 데이터
        console.log("Delete link", d);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        // React 상태에서 해당 Edge 삭제
        setGraphData((prevData) => ({
          ...prevData,
          links: prevData.links.filter((link) => link !== d), // 선택된 Edge 삭제
        }));
      });
    }

    // Update node and edge positions
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

      nodeGroup.attr("transform", (d) => {
        d.x = Math.max(20, Math.min(width - 20, d.x)); // Keep within width
        d.y = Math.max(20, Math.min(height - 20, d.y)); // Keep within height
        return `translate(${d.x}, ${d.y})`;
      });
    });

    // Add node on canvas click in edit mode
    svg.on("click", (event) => {
      console.log("SVG Clicked", event);
      if (currentMode === "draw") {
        const svgElement = svgRef.current;
        const point = svgElement.createSVGPoint();
        point.x = event.clientX; // 마우스 스크린 X 좌표
        point.y = event.clientY; // 마우스 스크린 Y 좌표

        // 스크린 좌표를 SVG 좌표로 변환
        const transformedPoint = point.matrixTransform(
          svgElement.getScreenCTM().inverse()
        );

        setInputPosition({ x: transformedPoint.x, y: transformedPoint.y }); // 정확한 SVG 좌표로 입력창 위치 설정
      }
    });

    // Function to re-render graph
  }, [tempEdge, graphData, setGraphData, currentMode]);

  // Handle input submit
  const handleInputSubmit = (event) => {
    event.preventDefault();
    if (inputValue.trim()) {
      const newNode = {
        id: `node-${Date.now()}`, // Unique ID
        name: inputValue.trim(), // User-provided name
        type: selectedType, // User-selected type
        x: inputPosition.x,
        y: inputPosition.y,
      };
      setGraphData((prevData) => ({
        ...prevData,
        nodes: [...prevData.nodes, newNode], // Add new node
      }));
      setInputValue(""); // Reset input
      setSelectedType("object"); // Reset type
      setInputPosition(null); // Hide input
    }
  };

  return (
    <div>
      <svg ref={svgRef}></svg>
      <style>{`
        .link {
          stroke: #aaa;
          stroke-width: 2;
        }
      `}</style>
      {/* Node name input */}
      {inputPosition && currentMode === "draw" && (
        <form
          onSubmit={handleInputSubmit}
          style={{
            position: "absolute",
            left: inputPosition.x,
            top: inputPosition.y,
            transform: "translate(-50%, -50%)",
            background: "white",
            border: "1px solid gray",
            padding: "10px",
            borderRadius: "5px",
            boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
          }}
        >
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter node name"
            style={{ marginBottom: "10px", width: "150px", padding: "5px" }}
            autoFocus
          />
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            style={{ marginBottom: "10px", width: "150px", padding: "5px" }}
          >
            <option value="object">Object</option>
            <option value="attribute">Attribute</option>
            <option value="relationship">Relationship</option>
          </select>
          <button
            type="submit"
            style={{
              width: "100%",
              padding: "5px",
              backgroundColor: "#4CAF50",
              color: "white",
              border: "none",
              cursor: "pointer",
            }}
          >
            Add Node
          </button>
        </form>
      )}
    </div>
  );
};

export default SceneGraph;
