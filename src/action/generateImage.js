const generateImage = async (sceneGraph) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/generate", {
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

const editImageWithMask = async (image_metadata, scene_graph, graph_changes) => { 
  try {
    const response = await fetch("http://127.0.0.1:5000/edit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      mode: "cors", // CORS 모드 명시
      body: JSON.stringify({
        image_metadata: image_metadata,
        scene_graph: scene_graph,
        graph_changes: graph_changes,
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

export { generateImage, editImageWithMask };