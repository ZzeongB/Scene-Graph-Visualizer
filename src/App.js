import React, { useState } from "react";
import SceneGraph from "./SceneGraph";
import generateSceneGraph from "./action/generateSceneGraph"; // 비동기 함수 가져오기

const App = () => {
  const [graphData, setGraphData] = useState(null); // Scene Graph 데이터 상태
  const [inputText, setInputText] = useState(""); // 사용자 입력 상태
  const [loading, setLoading] = useState(false); // 로딩 상태 관리

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true); // 로딩 시작
    try {
      const data = await generateSceneGraph(inputText); // 비동기 함수 호출
      setGraphData(data); // 데이터를 상태에 저장
    } catch (error) {
      console.error("Error generating scene graph:", error);
    }
    setLoading(false); // 로딩 종료
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Scene Graph Generator</h1>
      <form onSubmit={handleSubmit}>
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
      </form>
      {loading && <p>Loading...</p>}
      {graphData ? (
        <SceneGraph graphData={graphData} setGraphData={setGraphData}/>
      ) : (
        <p>Enter a prompt to generate a scene graph.</p>
      )}
    </div>
  );
};

export default App;
