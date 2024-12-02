import React, { useRef, useEffect } from "react";
import * as d3 from "d3";

const SceneGraph = ({ graphData }) => {
  const svgRef = useRef();

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
        node.fx = width / 2 - (objectNodes.length - 1) * 75 + objectIndex * 150; // x 좌표: object 노드 중심 정렬
        node.fy = height / 2; // y 좌표: 화면 중앙
      } else {
        node.fx = null; // 다른 노드는 움직일 수 있도록 초기화
        node.fy = null;
      }
    });

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    svg.selectAll("*").remove();

    const simulation = d3
      .forceSimulation(graphData.nodes)
      .force(
        "link",
        d3
          .forceLink(graphData.links)
          .id((d) => d.id)
          .distance(150) // 링크 거리 증가
      )
      .force("charge", d3.forceManyBody().strength(-300)) // 반발력 줄임
    //   .force("x", d3.forceX(width / 2).strength(0.1))
    //   .force("y", d3.forceY(height / 2).strength(0.1));

    // 화살표 끝점 띄우기
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 9) // 화살표 끝점을 노드에서 더 띄움
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 5) // 화살표 크기 조정
      .attr("markerHeight", 5)
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#aaa");

    // 링크 추가
    const link = svg
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graphData.links)
      .join("line")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");

    // 노드 추가
    const nodeGroup = svg
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(graphData.nodes)
      .join("g")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // 사각형 노드
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
      .attr("rx", 10) // 모서리 둥글게
      .attr("ry", 10);

    // 텍스트 노드
    nodeGroup
      .append("text")
      .attr("text-anchor", "middle")
      .attr("alignment-baseline", "middle")
      .attr("fill", "#fff")
      .style("font-size", "12px")
      .text((d) => d.name);

    // 엣지 길이 조정
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return d.source.x + (dx / distance) * 20; // 20px 띄움
        })
        .attr("y1", (d) => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return d.source.y + (dy / distance) * 20; // 20px 띄움
        })
        .attr("x2", (d) => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return d.target.x - (dx / distance) * 20; // 20px 띄움
        })
        .attr("y2", (d) => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return d.target.y - (dy / distance) * 20; // 20px 띄움
        });

      nodeGroup.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
    });
  }, [graphData]);

  return <svg ref={svgRef}></svg>;
};

export default SceneGraph;
