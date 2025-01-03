const generateSceneGraph = async (prompt) => {
  const apiKey = process.env.REACT_APP_OPENAI_API_KEY; // API 키 가져오기
  const endpoint = "https://api.openai.com/v1/chat/completions";

  const sceneGraphPrompt = `
    For the provided text prompt, generate a Scene Graph in JSON format.
    Include the following:
    1. Objects (relevant entities in the scene).
    2. Object attributes (such as color, size, position).
    3. Object relationships (e.g., on, under, next to).

    Example Output:
    {
      "objects": [
        { "id": "object1", "name": "cat", "attributes": ["sitting", "white"] },
        { "id": "object2", "name": "table", "attributes": ["wooden", "brown"] }
      ],
      "relationships": [
        { "source": "object1", "target": "object2", "relation": "on" }
      ]
    }

    There are following rules.
    1. Relationships are represented as an array of objects with source, target, and relation.
    2. Each word in the prompt must be classified as ONLY ONE of the following: object, attribute, or relationship. NO DUPLICATES.

    For example, for the given prompt "wolf holding chocolate icecream with strawberry on top",
    the scene graph would be:
    {
      "objects": [
        { "id": "object1", "name": "wolf" }
        { "id": "object2", "name": "icecream", "attributes": ["chocolate"] },
        { "id": "object3", "name": "strawberry" }
      ],
      "relationships": [
        { "source": "object3", "target": "object2", "relation": "on top" }
        { "source": "object1", "target": "object2", "relation": "holding" }
      ]
    }


    Use the prompt: "${prompt}".
    `;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini", // GPT 모델
        messages: [
          {
            role: "user",
            content: sceneGraphPrompt,
          },
        ],
        max_tokens: 512,
        temperature: 0,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      const sceneGraph = JSON.parse(data.choices[0].message.content);
      // console.log("Scene Graph (Raw):", sceneGraph);

      // 변환된 데이터로 그래프 구성
      const transformedGraph = transformGraphData(sceneGraph);

      // console.log("Scene Graph (Transformed):", transformedGraph);
      // return transformedGraph;

      return sceneGraph;
    } else {
      console.error("Error:", data);
      alert("Failed to generate scene graph. Check console for details.");
    }
  } catch (error) {
    console.error("Error fetching from OpenAI:", error);
    alert("An error occurred while generating the scene graph.");
  }
};

const transformGraphData = (sceneGraph) => {
   
  if (!sceneGraph || !sceneGraph.objects || !sceneGraph.relationships) {
    console.error("Invalid sceneGraph structure:", sceneGraph);
    return { nodes: [], links: [] };
  }

  // 노드 생성
  const nodes = [
    ...sceneGraph.objects.map((obj) => ({
      id: obj.id,
      name: obj.name,
      type: "object",
    })),
    ...sceneGraph.objects
      .flatMap((obj) =>
        (obj.attributes || []).map((attr) => ({
          id: `${obj.id}-${attr}`,
          name: attr,
          type: "attribute",
        }))
      ),
    ...sceneGraph.relationships.map((rel, index) => ({
      id: `rel-${index}`,
      name: rel.relation,
      type: "relationship",
    })),
  ];

  // 링크 생성
  const links = [
    ...sceneGraph.relationships.map((rel, index) => ({
      source: rel.source,
      target: `rel-${index}`, // 관계 노드로 연결
      relation: rel.relation,
    })),
    ...sceneGraph.relationships.map((rel, index) => ({
      source: `rel-${index}`, // 관계 노드에서 대상 노드로 연결
      target: rel.target,
      relation: rel.relation,
    })),
    ...sceneGraph.objects.flatMap((obj) =>
      (obj.attributes || []).map((attr) => ({
        source: obj.id,
        target: `${obj.id}-${attr}`,
        relation: "has attribute",
      }))
    ),
  ];

  return { nodes, links };
};

export { generateSceneGraph, transformGraphData };