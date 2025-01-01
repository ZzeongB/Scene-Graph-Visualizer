import React, { useEffect, useState } from "react";
import SceneGraph from "./SceneGraph";
import generateSceneGraph from "./action/generateSceneGraph"; // 씬 그래프 생성 함수
import generateUpdatedTextUsingAPI from "./action/generateUpdatedText";
import generateImage from "./action/generateImage";

const App = () => {
  const [sceneGraph, setSceneGraph] = useState({
    // sceneGraph is in the form of a JSON object, with {"objects":["attributes"], "relationships"}
    objects: [
      { id: "object1", name: "wolf" },
      { id: "object2", name: "icecream", attributes: ["chocolate"] },
    ],
    relationships: [
      { source: "object1", target: "object2", relation: "holding" },
    ],
  });
  const [graphData, setGraphData] = useState({
    // graphData is in the form of a JSON object, with {"nodes", "links"}, to be easily used by the SceneGraph component
    nodes: [
      { id: "object1", name: "wolf", type: "object" },
      { id: "object2", name: "icecream", type: "object" },
      { id: "object2-chocolate", name: "chocolate", type: "attribute" },
      { id: "rel-0", name: "holding", type: "relationship" },
    ],
    links: [
      { source: "object1", target: "rel-0", relation: "holding" },
      { source: "rel-0", target: "object2", relation: "holding" },
      {
        source: "object2",
        target: "object2-chocolate",
        relation: "has attribute",
      },
    ],
  }); // 씬 그래프 데이터 상태
  const [image, setImage] = useState(null);

  const [inputText, setInputText] = useState("wolf holding chocolate icecream"); // 사용자 입력 상태
  const [loading, setLoading] = useState(false); // 로딩 상태 관리
  const [currentMode, setCurrentMode] = useState("default"); // "default", "edit", "custom"

  useEffect(() => {
    // 초기 로딩 시 씬 그래프 생성
    console.log("image generated", image);
  }, [image]);
  // 모드 변경 함수
  const changeMode = (mode) => {
    setCurrentMode(mode);
  };

  // 씬 그래프 생성 함수
  const handleGenerateSceneGraph = async (event) => {
    event.preventDefault();
    setLoading(true); // 로딩 시작
    try {
      const data = await generateSceneGraph(inputText); // 씬 그래프 생성
      const { sceneGraph, nodes, links } = data;
      setSceneGraph(sceneGraph); // 생성된 씬 그래프 설정
      setGraphData({ nodes, links }); // 생성된 그래프 데이터 설정
    } catch (error) {
      console.error("Error generating scene graph:", error);
    }
    setLoading(false); // 로딩 종료
  };

  // 씬 그래프를 텍스트로 변환하는 함수
  const handleGenerateTextFromSceneGraph = async () => {
    setLoading(true); // 로딩 시작
    try {
      // API를 호출하여 새로운 텍스트 생성
      const newText = await generateUpdatedTextUsingAPI(graphData, inputText);

      console.log("Updated Text in App.js:", newText);
      setInputText(newText); // 새로 생성된 텍스트 업데이트
    } catch (error) {
      console.error("Error generating updated text:", error);
    }
    setLoading(false); // 로딩 종료
  };

  const handleGenerateImageSubmit = async () => {
    setLoading(true); // 로딩 시작
    try {
      const result = await generateImage(sceneGraph);

      setImage(result["image"]);
      console.log("Image generated successfully");
    } catch (error) {
      console.error("Error generating image:", error);
    }
    setLoading(false); // 로딩 종료
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Scene Graph Generator</h1>
      <div
        className="input-container"
        style={{ display: "flex", flexDirection: "row" }}
      >
        <form>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter a text prompt"
            style={{ width: "400px", padding: "10px", marginRight: "10px" }}
          />
        </form>
        <button
          type="submit"
          style={{ padding: "10px" }}
          onClick={handleGenerateSceneGraph}
        >
          Generate
        </button>
        <button
          onClick={handleGenerateTextFromSceneGraph}
          style={{ padding: "10px" }}
        >
          Convert Scene Graph to Text
        </button>
        <button onClick={handleGenerateImageSubmit} style={{ padding: "10px" }}>
          GenerateImage
        </button>
      </div>

      <div className="mode-selector">
        <button onClick={() => changeMode("default")}>Default</button>
        <button onClick={() => changeMode("draw")}>Draw</button>
        <button onClick={() => changeMode("edit")}>Edit</button>
        <button onClick={() => changeMode("delete")}>Delete</button>
      </div>

      {/* 현재 모드 표시 */}
      <p style={{ fontWeight: "bold", marginTop: "20px" }}>
        Current Mode: <span style={{ color: "blue" }}>{currentMode}</span>
      </p>

      {loading && <p>Loading...</p>}
      <div style={{ display: "flex", flexDirection: "row" }}>
        {graphData ? (
          <SceneGraph
            graphData={graphData}
            setGraphData={setGraphData}
            currentMode={currentMode}
            inputText={inputText}
            setInputText={setInputText}
          />
        ) : (
          <p>Enter a prompt to generate a scene graph.</p>
        )}

        <div>
          {image && (
            <img src={`data:image/png;base64,${image}`} alt="Generated" />
          )}
          {!image && <p>No image generated yet.</p>}
        </div>
      </div>
    </div>
  );
};

export default App;
