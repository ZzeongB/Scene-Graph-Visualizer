import React, { useEffect, useRef, useState } from 'react';

const ImageViewer = ({ image, masks = [] }) => {
  const [hoveredMask, setHoveredMask] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // 원본 이미지 로드
    const baseImage = new Image();
    baseImage.onload = () => {
      // 캔버스 크기를 이미지 크기로 설정
      canvas.width = baseImage.width;
      canvas.height = baseImage.height;
      
      // 이미지 그리기
      ctx.drawImage(baseImage, 0, 0);

      // 현재 hover된 마스크가 있다면 그리기
      if (hoveredMask) {
        const maskImage = new Image();
        maskImage.onload = () => {
          ctx.globalAlpha = 0.3;
          ctx.fillStyle = '#0066ff';
          ctx.drawImage(maskImage, 0, 0);
        };
        maskImage.src = `data:image/png;base64,${hoveredMask.mask}`;
      }
    };
    baseImage.src = `data:image/png;base64,${image}`;
  }, [image, hoveredMask]);

  const handleMouseMove = (e) => {
    if (!masks.length) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // 마스크 확인
    const checkMask = async (maskData) => {
      return new Promise((resolve) => {
        const maskImage = new Image();
        maskImage.onload = () => {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = canvas.width;
          tempCanvas.height = canvas.height;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(maskImage, 0, 0);

          const pixel = tempCtx.getImageData(x * scaleX, y * scaleY, 1, 1).data;
          resolve(pixel[0] === 255);
        };
        maskImage.src = `data:image/png;base64,${maskData.mask}`;
      });
    };

    const findHoveredMask = async () => {
      for (const maskData of masks) {
        if (await checkMask(maskData)) {
          setHoveredMask(maskData);
          return;
        }
      }
      setHoveredMask(null);
    };

    findHoveredMask();
  };

  return (
    <div className="relative inline-block">
      <canvas
        ref={canvasRef}
        className="max-w-full h-auto"
        onMouseMove={handleMouseMove}
      />
      {hoveredMask && (
        <div className="absolute bottom-0 left-0 bg-black bg-opacity-50 text-white px-2 py-1">
          {hoveredMask.name}
        </div>
      )}
    </div>
  );
};

export default ImageViewer;