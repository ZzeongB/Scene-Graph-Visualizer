import React, { useEffect, useRef, useState } from 'react';

const ImageViewer = ({ image, masks = [] }) => {
  const [hoveredMask, setHoveredMask] = useState(null);
  const [selectedMask, setSelectedMask] = useState(null);
  const bgCanvasRef = useRef(null);
  const fgCanvasRef = useRef(null);
  const topCanvasRef = useRef(null);
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

      // 최상위 캔버스 설정
      const topCanvas = topCanvasRef.current;
      topCanvas.width = img.width;
      topCanvas.height = img.height;
    };
    img.src = `data:image/png;base64,${image}`;
  }, [image]);

  const drawMaskedOriginal = async (maskData) => {
    if (!maskData || !image) return;

    console.log('Draw masked original:', maskData.name);
    
    const canvas = topCanvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [originalImg, maskImg] = await Promise.all([
      new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = `data:image/png;base64,${image}`;
      }),
      new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = `data:image/png;base64,${maskData.mask}`;
      })
    ]);

    // 원본 이미지를 먼저 그립니다
    ctx.drawImage(originalImg, 0, 0);

    // 마스크 처리를 위한 임시 캔버스
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(maskImg, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const maskImageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const maskPixels = maskImageData.data;

    for (let i = 0; i < data.length; i += 4) {
      const maskR = maskPixels[i];
      const maskG = maskPixels[i + 1];
      const maskB = maskPixels[i + 2];

      // 마스크의 검은색 부분을 투명하게 만듭니다
      if (maskR <= 5 && maskG <= 5 && maskB <= 5) {
        data[i + 3] = 0;  // 알파 채널을 0으로 설정
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const drawMask = async (maskData, isPreview = true, isSelected = false) => {
    if (!maskData) return;

    const canvas = fgCanvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const maskImg = new Image();
    maskImg.onload = () => {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = maskImg.width;
      tempCanvas.height = maskImg.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(maskImg, 0, 0);

      const imageData = tempCtx.getImageData(0, 0, maskImg.width, maskImg.height);
      const data = imageData.data;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        if (r <= 5 && g <= 5 && b <= 5) {
          data[i] = 50;
          data[i + 1] = 50;
          data[i + 2] = 50;
          data[i + 3] = isPreview ? 76 : 204;
        } else if (r >= 250 && g >= 250 && b >= 250) {
          if (isSelected) {
            data[i] = 50;
            data[i + 1] = 50;
            data[i + 2] = 50;
            data[i + 3] = 204;
          } else {
            data[i + 3] = 0;
          }
        }
      }

      ctx.globalCompositeOperation = 'copy';
      ctx.putImageData(imageData, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
    };

    maskImg.src = `data:image/png;base64,${maskData.mask}`;
  };

  const handleMouseMove = (e) => {
    const canvas = fgCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

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
        const ctx = fgCanvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        const topCtx = topCanvasRef.current.getContext('2d');
        topCtx.clearRect(0, 0, topCtx.canvas.width, topCtx.canvas.height);
      }
    };

    findHoveredMask();
  };

  const handleClick = async () => {
    if (hoveredMask) {
      if (selectedMask?.name === hoveredMask.name) {
        console.log('Deselected:', hoveredMask.name);
        setSelectedMask(null);
        drawMask(hoveredMask, true, false);
        const topCtx = topCanvasRef.current.getContext('2d');
        topCtx.clearRect(0, 0, topCtx.canvas.width, topCtx.canvas.height);
      } else {
        console.log('Selected:', hoveredMask.name);
        setSelectedMask(hoveredMask);
        drawMask(hoveredMask, false, true);
        await drawMaskedOriginal(hoveredMask);
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
      <canvas
        ref={topCanvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
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