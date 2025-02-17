import { clear } from "@testing-library/user-event/dist/clear";
import React, { useEffect, useRef, useState } from "react";

const loadImage = (src) =>
  new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = `data:image/png;base64,${src}`;
  });

const clearCanvas = (canvas) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
};

const ImageViewer = ({
  image,
  masks = [],
  graphData,
  setGraphData,
  sceneGraph,
  setSceneGraph,
  dispatch,
}) => {
  const [hoveredMask, setHoveredMask] = useState(null);
  const [selectedMask, setSelectedMask] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const bgCanvasRef = useRef(null);
  const fgCanvasRef = useRef(null);
  const topCanvasRef = useRef(null);

  useEffect(() => {
    if (!image) return;
    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height });
      [bgCanvasRef, fgCanvasRef, topCanvasRef].forEach((ref) => {
        if (ref.current) {
          ref.current.width = img.width;
          ref.current.height = img.height;
        }
      });
      bgCanvasRef.current.getContext("2d").drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${image}`;
  }, [image]);

  const drawMaskedOriginal = async (maskData) => {
    if (!maskData || !image) return;
    const canvas = topCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [originalImg, maskImg] = await Promise.all([
      loadImage(image),
      loadImage(maskData.mask),
    ]);

    // 원본 이미지를 먼저 그립니다
    ctx.drawImage(originalImg, 0, 0);

    // 마스크 처리를 위한 임시 캔버스
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(maskImg, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const maskImageData = tempCtx.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    );
    const data = imageData.data;
    const maskPixels = maskImageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const [maskR, maskG, maskB] = [
        maskPixels[i],
        maskPixels[i + 1],
        maskPixels[i + 2],
      ];
      const isBlack = maskR <= 5 && maskG <= 5 && maskB <= 5;
      if (isBlack) data[i + 3] = 0; // 알파 채널을 0으로 설정
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const drawMask = async (maskData, isPreview = true, isSelected = false) => {
    if (!maskData) return;

    const canvas = fgCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const maskImg = new Image();
    maskImg.src = `data:image/png;base64,${maskData.mask}`;
    await maskImg.decode();

    const tempCanvas = document.createElement("canvas");
    Object.assign(tempCanvas, { width: maskImg.width, height: maskImg.height });
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(maskImg, 0, 0);

    const imageData = tempCtx.getImageData(0, 0, maskImg.width, maskImg.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
      const isBlack = r <= 5 && g <= 5 && b <= 5;
      const isWhite = r >= 250 && g >= 250 && b >= 250;

      if (isBlack || (isWhite && isSelected)) {
        [data[i], data[i + 1], data[i + 2], data[i + 3]] = [
          50,
          50,
          50,
          isPreview ? 76 : 204,
        ];
      } else if (isWhite) {
        data[i + 3] = 0;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const handleMouseMove = (e) => {
    const canvas = fgCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    const checkMask = async (maskData) => {
      const maskImg = new Image();
      await new Promise((resolve) => {
        maskImg.onload = resolve;
        maskImg.src = `data:image/png;base64,${maskData.mask}`;
      });

      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext("2d");
      tempCtx.drawImage(maskImg, 0, 0);

      const pixel = tempCtx.getImageData(x, y, 1, 1).data;
      return pixel[0] === 255;
    };

    const findHoveredMask = async () => {
      for (const maskData of masks) {
        if (await checkMask(maskData)) {
          if (hoveredMask?.name !== maskData.name) {
            setHoveredMask(maskData);
            if (!selectedMask) {
              drawMask(maskData, true, false);
            }
          }
          return;
        }
      }

      if (hoveredMask && !selectedMask) {
        setHoveredMask(null);
        clearCanvas(fgCanvasRef.current);
        clearCanvas(topCanvasRef.current);
      }
    };

    findHoveredMask();
  };

  const handleClick = async () => {
    if (hoveredMask) {
      if (selectedMask?.name === hoveredMask.name) {
        setSelectedMask(null);
        drawMask(hoveredMask, true, false);
        clearCanvas(topCanvasRef.current);
      } else {
        setSelectedMask(hoveredMask);
        drawMask(hoveredMask, false, true);
        await drawMaskedOriginal(hoveredMask);
      }
    }
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMoveTop = (e) => {
    if (!isDragging) return;
    console.log("Dragging", e.clientX, dragStart.x, e.clientY, dragStart.y);
    setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const addNode = () => {
    // Add Node to Scene Graph
    // 1. Add a new node to GraphData
    // 2. Add a new edge to GraphData
    // 3. Update Scene Graph
    // 4. Update Graph Changes
    let [move_x, move_y] = [position.x, position.y];
    const attributeName = "MOVE";

    const objNode = graphData.nodes.find(
      (node) => node.type === "object" && node.name === selectedMask?.name
    );

    // If object already has MOVE attribute, update existing node
    const existingNode = graphData.nodes.find(
      (node) =>
        node.type === "attribute" &&
        node.name === attributeName &&
        node.id.includes(objNode.id)
    );
    let updatedGraphData = graphData;
    let updatedSceneGraph = sceneGraph;
    let updatedGraphChanges = {};

    if (existingNode) {
      console.log("Update Existing Node", existingNode);
      let [existing_x, existing_y] = existingNode.id.split("_").slice(-2); // change to number
      existing_x = parseInt(existing_x);
      existing_y = parseInt(existing_y);
      console.log(
        "Existing Node Position",
        existing_x,
        existing_y,
        move_x,
        move_y
      );
      // [move_x, move_y] = [move_x + existing_x, move_y + existing_y];
      const newNode = {
        id: `${objNode.id}-${attributeName}_${move_x}_${move_y}`, // Update position
        name: attributeName, // User-provided name
        type: "attribute", // User-selected type
        x: objNode.x, // Default position
        y: objNode.y + 100, // Default position
      };

      const newEdge = {
        source: objNode.id, // Source ID
        target: newNode.id, // Target ID
        relation: "has attribute", // User-provided relation
      };

      updatedGraphData = {
        nodes: graphData.nodes.map((node) =>
          node.id === existingNode.id ? newNode : node
        ),
        links: graphData.links.map((link) =>
          link.source.id === objNode.id && link.target.id === existingNode.id
            ? newEdge
            : link
        ),
      };

      updatedGraphChanges = {
        type: "add attribute",
        object: objNode.id,
        attribute: `attributeName_${move_x}_${move_y}`,
      };
    } else {
      const newNode = {
        id: `${objNode.id}-${attributeName}_${move_x}_${move_y}`, // Unique ID
        name: attributeName, // User-provided name
        type: "attribute", // User-selected type
        x: objNode.x, // Default position
        y: objNode.y + 100, // Default position
      };

      console.log("Add New Node", newNode);

      const newEdge = {
        source: objNode.id, // Source ID
        target: newNode.id, // Target ID
        relation: "has attribute", // User-provided relation
      };

      updatedGraphData = {
        nodes: [...graphData.nodes, newNode],
        links: [...graphData.links, newEdge],
      };

      updatedSceneGraph = {
        ...sceneGraph,
        objects: sceneGraph.objects.map((obj) =>
          obj.id === objNode.id
            ? {
                ...obj,
                attributes: obj.attributes
                  ? [...obj.attributes, newNode.name]
                  : [newNode.name],
              }
            : obj
        ),
      };

      updatedGraphChanges = {
        type: "add attribute",
        object: objNode.id,
        attribute: `attributeName_${move_x}_${move_y}`,
      };
    }

    setGraphData(updatedGraphData);
    setSceneGraph(updatedSceneGraph);
    dispatch({
      type: "ADD_CHANGE",
      payload: updatedGraphChanges,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    addNode();
  };
  const handleMouseLeave = () => setIsDragging(false);

  return (
    <div
      style={{
        position: "relative",
        width: imageSize.width,
        height: imageSize.height,
        display: "inline-block",
      }}
    >
      {[bgCanvasRef, fgCanvasRef, topCanvasRef].map((ref, index) => (
        <canvas
          key={index}
          ref={ref}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents:
              ref === fgCanvasRef
                ? selectedMask
                  ? "none"
                  : "auto"
                : ref == topCanvasRef
                ? selectedMask
                  ? "auto"
                  : "none"
                : "none",
            transform:
              ref === topCanvasRef
                ? `translate(${position.x}px, ${position.y}px)`
                : "none",
            cursor:
              ref === topCanvasRef
                ? isDragging
                  ? "grabbing"
                  : "grab"
                : "default",
          }}
          onMouseMove={
            ref === fgCanvasRef
              ? handleMouseMove
              : ref === topCanvasRef
              ? handleMouseMoveTop
              : undefined
          }
          onClick={ref === fgCanvasRef ? handleClick : undefined}
          onDoubleClick={ref === topCanvasRef ? handleClick : undefined}
          onMouseDown={ref === topCanvasRef ? handleMouseDown : undefined}
          onMouseUp={ref === topCanvasRef ? handleMouseUp : undefined}
          onMouseLeave={ref === topCanvasRef ? handleMouseLeave : undefined}
        />
      ))}
      {hoveredMask && (
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            background: "rgba(0,0,0,0.5)",
            color: "white",
            padding: "4px 8px",
          }}
        >
          {hoveredMask.name}
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
