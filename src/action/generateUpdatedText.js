const generateUpdatedTextUsingAPI = async (sceneGraph, beforeText) => {
  const apiKey = process.env.REACT_APP_OPENAI_API_KEY; // OpenAI API 키 가져오기
  const endpoint = "https://api.openai.com/v1/chat/completions";

  const prompt = `
    Here is a scene description and a scene graph update. Your task is to generate a new scene description that incorporates the updates in the scene graph while preserving the style and structure of the original description.

    ### Example
    Original description:
    "A cat is sitting on a wooden table."

    Scene graph update:
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

    ### Your task
    Original description:
    "${beforeText}"

    Scene graph update:
    ${JSON.stringify(sceneGraph, null, 2)}

    Please provide the updated scene description:
  `;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4",
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
    alert("An error occurred while generating the updated text.");
    return beforeText; // 에러 시 기존 텍스트 반환
  }
};

export default generateUpdatedTextUsingAPI;
