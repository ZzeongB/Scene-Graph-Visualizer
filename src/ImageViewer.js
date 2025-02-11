import React, { useEffect, useRef, useState } from 'react';

const ImageViewer = ({ image, masks = [] }) => {
  const [hoveredMask, setHoveredMask] = useState(null);
  const [selectedMask, setSelectedMask] = useState(null);
  const bgCanvasRef = useRef(null);
  const fgCanvasRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  // 이미지 로드 및 초기 설정
  useEffect(() => {
    if (!image) return;

    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height });
      
      // 배경 캔버스 설정
      const bgCanvas = bgCanvasRef.current;
      bgCanvas.width = img.width;
      bgCanvas.height = img.height;
      const bgCtx = bgCanvas.getContext('2d');
      bgCtx.drawImage(img, 0, 0);

      // 전경 캔버스 설정
      const fgCanvas = fgCanvasRef.current;
      fgCanvas.width = img.width;
      fgCanvas.height = img.height;
    };
    img.src = `data:image/png;base64,${image}`;
  }, [image]);

  // 마스크 그리기
  const drawMask = async (maskData, isPreview = true) => {
    if (!maskData) return;

    const canvas = fgCanvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const maskImg = new Image();
    maskImg.onload = () => {
      ctx.save();
      ctx.globalAlpha = isPreview ? 0.3 : 0.8;
      ctx.fillStyle = '#0066ff';
      ctx.drawImage(maskImg, 0, 0);
      ctx.restore();
    };
    maskImg.src = `data:image/png;base64,${maskData.mask}`;
  };

  const handleMouseMove = (e) => {
    const canvas = fgCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // 마스크 감지
    const checkMask = async (maskData) => {
      const maskImg = new Image();
      await new Promise(resolve => {
        maskImg.onload = resolve;
        maskImg.src = `data:image/png;base64,${maskData.mask}`;
      });

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(maskImg, 0, 0);

      const pixel = tempCtx.getImageData(x, y, 1, 1).data;
      return pixel[0] === 255;
    };

    const findHoveredMask = async () => {
      for (const maskData of masks) {
        if (await checkMask(maskData)) {
          if (hoveredMask?.name !== maskData.name) {
            console.log('Hover detected:', maskData.name);
            setHoveredMask(maskData);
            if (!selectedMask) {
              drawMask(maskData, true);
            }
          }
          return;
        }
      }

      if (hoveredMask && !selectedMask) {
        console.log('Hover removed');
        setHoveredMask(null);
        const ctx = fgCanvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    };

    findHoveredMask();
  };

  const handleClick = () => {
    if (hoveredMask) {
      if (selectedMask?.name === hoveredMask.name) {
        console.log('Deselected:', hoveredMask.name);
        setSelectedMask(null);
        drawMask(hoveredMask, true);
      } else {
        console.log('Selected:', hoveredMask.name);
        setSelectedMask(hoveredMask);
        drawMask(hoveredMask, false);
      }
    }
  };

  return (
    <div style={{ 
      position: 'relative', 
      width: imageSize.width, 
      height: imageSize.height,
      display: 'inline-block'
    }}>
      <canvas
        ref={bgCanvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%'
        }}
      />
      <canvas
        ref={fgCanvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%'
        }}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      />
      {hoveredMask && (
        <div 
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            background: 'rgba(0,0,0,0.5)',
            color: 'white',
            padding: '4px 8px'
          }}
        >
          {hoveredMask.name}
        </div>
      )}
    </div>
  );
};

export default ImageViewer;