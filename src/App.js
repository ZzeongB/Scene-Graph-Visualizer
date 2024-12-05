import React, { useState } from "react";
import SceneGraph from "./SceneGraph";
import generateSceneGraph from "./action/generateSceneGraph"; // 씬 그래프 생성 함수
import generateUpdatedTextUsingAPI from "./action/generateUpdatedText";

const App = () => {
  const [graphData, setGraphData] = useState({
    nodes: [
      { id: "object1", name: "wolf", type: "object" },
      { id: "object2", name: "chocolate icecream", type: "object" },
      { id: "rel-0", name: "holding", type: "relationship" },
    ],
    links: [
      { source: "object1", target: "rel-0", relation: "holding" },
      { source: "rel-0", target: "object2", relation: "holding" },
    ],
  }); // 씬 그래프 데이터 상태
  const [inputText, setInputText] = useState("wolf holding chocolate icecream"); // 사용자 입력 상태
  const [loading, setLoading] = useState(false); // 로딩 상태 관리
  const [currentMode, setCurrentMode] = useState("default"); // "default", "edit", "custom"

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
      setGraphData(data); // 생성된 그래프 데이터 설정
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

      console.log("Updated Text in SceneGraph.js:", newText);
      setInputText(newText); // 새로 생성된 텍스트 업데이트
    } catch (error) {
      console.error("Error generating updated text:", error);
    }
    setLoading(false); // 로딩 종료
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Scene Graph Generator</h1>
      <form onSubmit={handleGenerateSceneGraph}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter a text prompt"
          style={{ width: "400px", padding: "10px", marginRight: "10px" }}
        />
        <button type="submit" style={{ padding: "10px" }}>
          Generate
        </button>
        <button
          onClick={handleGenerateTextFromSceneGraph}
          style={{ padding: "10px", marginTop: "20px" }}
        >
          Convert Scene Graph to Text
        </button>
      </form>
      <div className="mode-selector">
        <button onClick={() => changeMode("default")}>Default Mode</button>
        <button onClick={() => changeMode("edit")}>Edit Mode</button>
        <button onClick={() => changeMode("custom")}>Custom Mode</button>
      </div>

      {/* 현재 모드 표시 */}
      <p style={{ fontWeight: "bold", marginTop: "20px" }}>
        Current Mode: <span style={{ color: "blue" }}>{currentMode}</span>
      </p>

      {loading && <p>Loading...</p>}
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
    </div>
  );
};

export default App;
