const generateUpdatedTextUsingAPI = async (sceneGraph, beforeText) => {
  const apiKey = process.env.REACT_APP_OPENAI_API_KEY; // OpenAI API 키 가져오기
  const endpoint = "https://api.openai.com/v1/chat/completions";

  const prompt = `
    Here is a prompt and a scene graph update. Your task is to generate a new prompt that incorporates the updates in the scene graph.

    ### Example
    Original prompt:
    "A cat is sitting on a wooden table."

    Updated scene graph:
    {
      "objects": [
        { "id": "object1", "name": "cat", "attributes": ["sitting", "white"] },
        { "id": "object2", "name": "table", "attributes": ["wooden", "brown"] },
        { "id": "object3", "name": "bowl", "attributes": ["red"] }
      ],
      "relationships": [
        { "source": "object1", "target": "object2", "relation": "on" },
        { "source": "object3", "target": "object2", "relation": "on" }
      ]
    }

    Updated description:
    "A white cat is sitting on a wooden table, and there is a red bowl on the table."

    There are following rules:
    1. The updated prompt should NEVER have other objects or attributes or relationships that are NOT IN THE SCENE GRAPH.
    2. The original description should be preserved as much as possible.
    3. You should avoid repeating the same information.

    ### Your task
    Original prompt:
    "${beforeText}"

    Scene graph update:
    ${JSON.stringify(sceneGraph, null, 2)}

    Please provide the updated scene prompt:
  `;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
        max_tokens: 200,
        temperature: 0.7,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      let updatedText = data.choices[0].message.content.trim();
      updatedText = updatedText.replace(/^"|"$/g, ""); // 앞뒤 큰따옴표 제거

      console.log("Generated Text:", updatedText);
      return updatedText;
    } else {
      console.error("Error:", data);
      alert("Failed to generate updated text. Check console for details.");
      return beforeText; // 실패 시 기존 텍스트 반환
    }
  } catch (error) {
    console.error("Error fetching from OpenAI:", error);
    return beforeText; // 에러 시 기존 텍스트 반환
  }
};

export default generateUpdatedTextUsingAPI;
