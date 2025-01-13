import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

const SceneGraph = ({
  graphData,
  setGraphData,
  currentMode,
  sceneGraph,
  setSceneGraph,
  dispatch,
  width = 800,
  height = 600,
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

    /// object, attribute, relationship 노드 필터링
    const objectNodes = graphData.nodes.filter(
      (node) => node.type === "object"
    );
    const attributeNodes = graphData.nodes.filter(
      (node) => node.type === "attribute"
    );
    const relationshipNodes = graphData.nodes.filter(
      (node) => node.type === "relationship"
    );

    // 노드 초기 위치 설정
    graphData.nodes.forEach((node) => {
      if (node.type === "object") {
        // object 노드는 중앙에 배치
        const objectIndex = objectNodes.findIndex(
          (objNode) => objNode.id === node.id
        );
        node.fx = width / 2 - (objectNodes.length - 1) * 75 + objectIndex * 150; // 가로 간격 150
        node.fy = height / 2; // 중앙
      } else if (node.type === "attribute") {
        // attribute 노드는 object 노드 근처에 배치
        const objectId = node.id.split("-")[0]; // object ID 추출
        const objectIndex = objectNodes.findIndex(
          (objNode) => objNode.id === objectId
        );

        // attribute 노드의 index를 계산
        const attributeIndex = attributeNodes
          .filter((attrNode) => attrNode.id.split("-")[0] === objectId)
          .findIndex((attrNode) => attrNode.id === node.id);

        // x축과 y축 모두 조정
        node.fx =
          width / 2 -
          (objectNodes.length - 1) * 75 +
          objectIndex * 150 +
          (attributeIndex % 2) * 75; // object와 같은 x축, 겹치지 않게 x축 조정
        node.fy = height / 2 - 100 - Math.floor(attributeIndex / 2) * 50; // object 위에 배치, 겹치지 않게 y축 조정
      } else if (node.type === "relationship") {
        // relationship 노드는 object와 object 사이에 배치
        const relationshipIndex = relationshipNodes.findIndex(
          (relNode) => relNode.id === node.id
        );
        node.fx =
          width / 2 -
          (relationshipNodes.length - 1) * 75 +
          relationshipIndex * 150; // 가로 간격 150
        node.fy = height / 2 + 100; // object 아래에 배치
      } else {
        // 기타 노드 초기화
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

              if (targetNode && targetNode !== draggingNodeRef.current) {
                // 새 엣지 추가
                const newEdge = {
                  source: draggingNodeRef.current,
                  target: targetNode,
                };

                // 'relation' of the edge
                // if attribute->object, "has attribute"
                // if object->relationship, name of the relationship
                // if relationship->object, name of the relationship
                // if neither, invalid
                let relation = "";

                if (
                  draggingNodeRef.current.type === "object" &&
                  targetNode.type === "attribute"
                ) {
                  relation = "has attribute";
                  newEdge.target.id = `${newEdge.source.id}-${newEdge.target.name}`;
                }

                if (
                  draggingNodeRef.current.type === "object" &&
                  targetNode.type === "relationship"
                ) {
                  relation = targetNode.name;
                }

                if (
                  draggingNodeRef.current.type === "relationship" &&
                  targetNode.type === "object"
                ) {
                  relation = draggingNodeRef.current.name;
                }

                if (!relation) {
                  console.warn("Invalid edge type");

                  setTempEdge(null);
                  draggingNodeRef.current = null;

                  return;
                }

                newEdge.relation = relation;

                // React 상태로 링크 업데이트
                let updatedGraphData = { ...graphData };
                updatedGraphData.links.push(newEdge);

                let updatedSceneGraph = { ...sceneGraph };
                let updatedGraphChanges = {};
                if (newEdge.relation === "has attribute") {
                  const objectId = newEdge.source.id;
                  const attributeName = newEdge.target.name;
                  updatedSceneGraph = {
                    ...sceneGraph,
                    objects: sceneGraph.objects.map((obj) =>
                      obj.id === objectId
                        ? {
                            ...obj,
                            attributes: obj.attributes
                              ? [...obj.attributes, attributeName]
                              : [attributeName],
                          }
                        : obj
                    ),
                  };
                  updatedGraphChanges = 
                    {
                      type: "add attribute", // 새로 추가할 객체의 내용
                      object: objectId,
                      attribute: attributeName,
                    };
                }
                if (newEdge.relation !== "has attribute") {
                  // then, relationship
                  // check if both links object->relationship->object exists
                  const existingRelationship = graphData.links.find(
                    (rel) =>
                      rel.relation === newEdge.relation &&
                      rel.source !== newEdge.source &&
                      rel.target !== newEdge.target
                  );

                  if (existingRelationship) {
                    // find the source and target objects whose type is "object"
                    let sourceId, targetId;
                    if (newEdge.source.type === "object") {
                      sourceId = newEdge.source.id;
                      targetId = existingRelationship.target.id;
                    } else {
                      sourceId = existingRelationship.source.id;
                      targetId = newEdge.target.id;
                    }

                    const newRelationship = {
                      source: sourceId,
                      target: targetId,
                      relation: newEdge.relation,
                    };

                    updatedSceneGraph = {
                      ...sceneGraph,
                      relationships: [
                        ...sceneGraph.relationships,
                        newRelationship,
                      ],
                    };
                    updatedGraphChanges = 
                      {
                        type: "add relationship",
                        source: sourceId,
                        target: targetId,
                        relation: newEdge.relation,
                      };
                    
                  }
                }

                setGraphData(updatedGraphData);
                setSceneGraph(updatedSceneGraph);
                dispatch({
                  type: "ADD_CHANGE",
                  payload: updatedGraphChanges,
                });
              }

              // Set SceneGraph

              // tempEdge 초기화
              setTempEdge(null);
              draggingNodeRef.current = null;
            }
          })
      );
    }

    if (currentMode === "edit") {
      nodeGroup.on("click", (event, node) => {
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        const newName = prompt("Enter new name for the node:", node.name);
        if (!newName || newName.trim() === "") {
          console.warn("Invalid name. Update aborted.");
          return;
        }

        // sceneGraph 업데이트
        let updatedSceneGraph = { ...sceneGraph };
        let updatedGraphChanges = {};
        let newId = node.id;

        if (node.type === "object") {
          updatedSceneGraph = {
            ...sceneGraph,
            objects: sceneGraph.objects.map((obj) =>
              obj.id === node.id ? { ...obj, name: newName } : obj
            ),
          };
          updatedGraphChanges = 
            { type: "rename object", object: node.id, name: newName }
          ;
        } else if (node.type === "attribute") {
          const objectId = node.id.split("-")[0];
          updatedSceneGraph = {
            ...sceneGraph,
            objects: sceneGraph.objects.map((obj) =>
              obj.id === objectId
                ? {
                    ...obj,
                    attributes: obj.attributes.map((attr) =>
                      attr === node.name ? newName : attr
                    ),
                  }
                : obj
            ),
          };
          updatedGraphChanges = 
            {
              type: "rename attribute",
              object: objectId,
              attribute: node.name,
              name: newName,
            }

          newId = `${objectId}-${newName}`;
        } else if (node.type === "relationship") {
          updatedSceneGraph = {
            ...sceneGraph,
            relationships: sceneGraph.relationships.map((rel) =>
              rel.relation === node.name ? { ...rel, relation: newName } : rel
            ),
          };
          updatedGraphChanges = 
            {
              type: "rename relationship",
              relation: node.name,
              name: newName,
            }
        }

        // graphData 업데이트
        const updatedGraphData = {
          links: graphData.links.map((link) => ({
            ...link,
            source: link.source.id === node.id ? newId : link.source,
            target: link.target.id === node.id ? newId : link.target,
          })),
          nodes: graphData.nodes.map((n) =>
            n.id === node.id ? { ...n, name: newName, id: newId } : n
          ),
        };

        // 상태 동기화
        setSceneGraph(updatedSceneGraph);
        setGraphData(updatedGraphData);
        dispatch({
          type: "ADD_CHANGE",
          payload: updatedGraphChanges,
        });
      });
    }

    if (currentMode === "delete") {
      nodeGroup.on("click", (event, node) => {
        console.log("Delete node", node);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        // React 상태에서 해당 노드 삭제
        let updatedGraphData = { ...graphData };
        updatedGraphData.nodes = updatedGraphData.nodes.filter(
          (n) => n !== node
        );
        updatedGraphData.links = updatedGraphData.links.filter(
          (link) => link.source !== node && link.target !== node
        );

        // SceneGraph 업데이트
        let updatedSceneGraph = { ...sceneGraph };
        let updatedGraphChanges = {};
        if (node.type === "object") {
          updatedSceneGraph = {
            ...sceneGraph,
            objects: sceneGraph.objects.filter((obj) => obj.id !== node.id),
            relationships: sceneGraph.relationships.filter(
              (rel) => rel.source !== node.id && rel.target !== node.id
            ),
          };

          updatedGraphChanges = 
            { type: "delete object", object: node.id }
        }
        if (node.type === "attribute") {
          const objectId = node.id.split("-")[0];
          updatedSceneGraph = {
            ...sceneGraph,
            objects: sceneGraph.objects.map((obj) =>
              obj.id === objectId
                ? {
                    ...obj,
                    attributes: obj.attributes.filter(
                      (attr) => attr !== node.name
                    ),
                  }
                : obj
            ),
          };
          updatedGraphChanges =
            {
              type: "delete attribute",
              object: objectId,
              attribute: node.name,
            }
        }
        if (node.type === "relationship") {
          updatedSceneGraph = {
            ...sceneGraph,
            relationships: sceneGraph.relationships.filter(
              (rel) => rel.relation !== node.name
            ),
          };
          updatedGraphChanges = 
            { type: "delete relationship", relation: node.name }
        }

        setGraphData(updatedGraphData);
        setSceneGraph(updatedSceneGraph);
        dispatch({
          type: "ADD_CHANGE",
          payload: updatedGraphChanges,
        });
      });

      link.on("click", (event, d) => {
        // 'd'는 해당 Edge 데이터
        console.log("Delete link", d);
        event.preventDefault(); // 기본 컨텍스트 메뉴 방지

        // React 상태에서 해당 Edge 삭제
        let updatedGraphData = { ...graphData };
        updatedGraphData.links = updatedGraphData.links.filter(
          (link) => link !== d
        );

        // SceneGraph 업데이트
        let updatedSceneGraph = { ...sceneGraph };
        let updatedGraphChanges = {};

        // if attribute->object, delete attribute from object
        if (d.relation === "has attribute") {
          const objectId = d.source.id;
          const attributeName = d.target.name;
          updatedSceneGraph = {
            ...sceneGraph,
            objects: sceneGraph.objects.map((obj) =>
              obj.id === objectId
                ? {
                    ...obj,
                    attributes: obj.attributes.filter(
                      (attr) => attr !== attributeName
                    ),
                  }
                : obj
            ),
          };
          updatedGraphChanges =
            {
              type: "delete attribute",
              object: objectId,
              attribute: attributeName,
            }
        }

        // if object->relationship, delete relationship
        if (d.source.type === "object" && d.target.type === "relationship") {
          updatedSceneGraph = {
            ...sceneGraph,
            relationships: sceneGraph.relationships.filter(
              (rel) => rel.relation !== d.relation
            ),
          };
          updatedGraphChanges = 
            {
              type: "delete relationship",
              source: d.source.id,
              target: d.target.id,
              relation: d.relation,
            }
        }
        if (d.source.type === "relationship" && d.target.type === "object") {
          updatedSceneGraph = {
            ...sceneGraph,
            relationships: sceneGraph.relationships.filter(
              (rel) => rel.relation !== d.relation
            ),
          };
          updatedGraphChanges =
            {
              type: "delete relationship",
              source: d.source.id,
              target: d.target.id,
              relation: d.relation,
            }
        }

        setGraphData(updatedGraphData);
        setSceneGraph(updatedSceneGraph);
        dispatch({
          type: "ADD_CHANGE",
          payload: updatedGraphChanges,
        });
      });
    }

    // Update node and edge positions
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => {
          if (!d.source || !d.target) return 0; // 기본값 처리
          const dx = (d.target.x ?? 0) - (d.source.x ?? 0);
          const dy = (d.target.y ?? 0) - (d.source.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 거리 0 방지
          return (d.source.x ?? 0) + (dx / distance) * 20; // 20px 띄움
        })
        .attr("y1", (d) => {
          if (!d.source || !d.target) return 0; // 기본값 처리
          const dx = (d.target.x ?? 0) - (d.source.x ?? 0);
          const dy = (d.target.y ?? 0) - (d.source.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 거리 0 방지
          return (d.source.y ?? 0) + (dy / distance) * 20; // 20px 띄움
        })
        .attr("x2", (d) => {
          if (!d.source || !d.target) return 0; // 기본값 처리
          const dx = (d.target.x ?? 0) - (d.source.x ?? 0);
          const dy = (d.target.y ?? 0) - (d.source.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 거리 0 방지
          return (d.target.x ?? 0) - (dx / distance) * 20; // 20px 띄움
        })
        .attr("y2", (d) => {
          if (!d.source || !d.target) return 0; // 기본값 처리
          const dx = (d.target.x ?? 0) - (d.source.x ?? 0);
          const dy = (d.target.y ?? 0) - (d.source.y ?? 0);
          const distance = Math.sqrt(dx * dx + dy * dy) || 1; // 거리 0 방지
          return (d.target.y ?? 0) - (dy / distance) * 20; // 20px 띄움
        });

      nodeGroup.attr("transform", (d) => {
        d.x = Math.max(20, Math.min(width - 20, d.x ?? width / 2)); // 기본값 지정
        d.y = Math.max(20, Math.min(height - 20, d.y ?? height / 2)); // 기본값 지정
        return `translate(${d.x}, ${d.y})`;
      });
    });

    // Add node on canvas click in edit mode
    svg.on("click", (event) => {
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
        id: `${selectedType + Date.now()}`, // Unique ID
        name: inputValue.trim(), // User-provided name
        type: selectedType, // User-selected type
        x: inputPosition.x,
        y: inputPosition.y,
      };
      setGraphData((prevData) => ({
        ...prevData,
        nodes: [...prevData.nodes, newNode], // Add new node
      }));

      // Update sceneGraph
      let updatedSceneGraph = { ...sceneGraph };
      let updatedGraphChanges = {};
      if (selectedType === "object") {
        updatedSceneGraph = {
          ...sceneGraph,
          objects: [
            ...sceneGraph.objects,
            { id: newNode.id, name: newNode.name },
          ],
        };
        updatedGraphChanges =
          { type: "add object", object: newNode.id, name: newNode.name }
      }
      setSceneGraph(updatedSceneGraph);
      dispatch({
        type: "ADD_CHANGE",
        payload: updatedGraphChanges,
      });
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
