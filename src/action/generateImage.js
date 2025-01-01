const generateImage = async (sceneGraph) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        mode: "cors", // CORS 모드 명시
        body: JSON.stringify({
          scene_graph: sceneGraph,
          size: 512,
        }),
      });
  
      if (!response.ok) {
        throw new Error("Error generating image");
      }
  
      const data = await response.json();
      console.log("response", data);
      return data;
    } catch (error) {
      console.error("Error generating image:", error);
      throw error;
    }
  };
  
  export default generateImage;