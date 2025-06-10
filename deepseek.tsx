import { type NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { messages, apiKey } = body

    // Validate API key
    if (!apiKey || typeof apiKey !== "string") {
      return NextResponse.json({ error: "API key is required" }, { status: 400 })
    }

    // Validate API key format (OpenRouter keys typically start with sk-or-v1-)
    if (!apiKey.startsWith("sk-or-v1-")) {
      return NextResponse.json(
        { error: "Invalid API key format. OpenRouter keys should start with 'sk-or-v1-'" },
        { status: 400 },
      )
    }

    // Validate messages
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json({ error: "Messages array is required and cannot be empty" }, { status: 400 })
    }

    // Validate message format
    for (const message of messages) {
      if (!message.role || !message.content) {
        return NextResponse.json({ error: "Each message must have 'role' and 'content' properties" }, { status: 400 })
      }
      if (!["user", "assistant", "system"].includes(message.role)) {
        return NextResponse.json({ error: "Message role must be 'user', 'assistant', or 'system'" }, { status: 400 })
      }
    }

    console.log("Making request to OpenRouter with model: deepseek/deepseek-r1")

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000",
        "X-Title": "DeepSeek Chat App",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "deepseek/deepseek-r1", // Updated model name
        messages: messages,
        stream: false,
        max_tokens: 4000,
        temperature: 0.7,
      }),
    })

    const responseText = await response.text()
    console.log("OpenRouter response status:", response.status)
    console.log("OpenRouter response:", responseText)

    if (!response.ok) {
      let errorMessage = "Failed to get response from DeepSeek"

      try {
        const errorData = JSON.parse(responseText)
        errorMessage = errorData.error?.message || errorData.message || errorMessage
        console.error("OpenRouter API error:", errorData)
      } catch (e) {
        console.error("OpenRouter API error (raw):", responseText)
      }

      if (response.status === 401) {
        return NextResponse.json(
          {
            error: "Invalid API key. Please check your OpenRouter API key and make sure it's active.",
          },
          { status: 401 },
        )
      }

      if (response.status === 429) {
        return NextResponse.json(
          {
            error: "Rate limit exceeded. Please try again later.",
          },
          { status: 429 },
        )
      }

      if (response.status === 400) {
        return NextResponse.json(
          {
            error: `Bad request: ${errorMessage}`,
          },
          { status: 400 },
        )
      }

      return NextResponse.json(
        {
          error: `API Error (${response.status}): ${errorMessage}`,
        },
        { status: response.status },
      )
    }

    let data
    try {
      data = JSON.parse(responseText)
    } catch (e) {
      console.error("Failed to parse OpenRouter response:", responseText)
      return NextResponse.json({ error: "Invalid response format from API" }, { status: 500 })
    }

    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      console.error("Invalid response structure:", data)
      return NextResponse.json({ error: "Invalid response format from API" }, { status: 500 })
    }

    return NextResponse.json({
      content: data.choices[0].message.content,
      usage: data.usage,
    })
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json(
      {
        error: `Internal server error: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 500 },
    )
  }
}
