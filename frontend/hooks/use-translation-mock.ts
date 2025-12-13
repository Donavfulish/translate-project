"use client"

import { useState } from "react"
import type { LanguageDirection } from "@/app/page"

// Mock translation data
const mockTranslations: Record<string, { en: string; vi: string }> = {
  "xin chào": { en: "Hello", vi: "xin chào" },
  "cảm ơn": { en: "Thank you", vi: "cảm ơn" },
  "tạm biệt": { en: "Goodbye", vi: "tạm biệt" },
  "bạn khỏe không": { en: "How are you?", vi: "bạn khỏe không" },
  "tôi yêu bạn": { en: "I love you", vi: "tôi yêu bạn" },
  hello: { en: "Hello", vi: "Xin chào" },
  "thank you": { en: "Thank you", vi: "Cảm ơn" },
  goodbye: { en: "Goodbye", vi: "Tạm biệt" },
  "how are you": { en: "How are you?", vi: "Bạn khỏe không?" },
  "i love you": { en: "I love you", vi: "Tôi yêu bạn" },
}

export function useTranslationMock() {
  const [isLoading, setIsLoading] = useState(false)

  const translate = async (text: string, direction: LanguageDirection): Promise<string> => {
    setIsLoading(true)

    // Simulate API delay (1.5 seconds)
    await new Promise((resolve) => setTimeout(resolve, 1500))

    const normalizedText = text.toLowerCase().trim()
    const targetLang = direction === "vi-to-en" ? "en" : "vi"

    // Check if we have a mock translation
    const mockEntry = mockTranslations[normalizedText]
    let result: string

    if (mockEntry) {
      result = mockEntry[targetLang]
    } else {
      // Return generic response for unknown inputs
      if (direction === "vi-to-en") {
        result = `Translation: "${text}" (This is a mock translation. In production, this would be translated by a Python backend.)`
      } else {
        result = `Bản dịch: "${text}" (Đây là bản dịch mô phỏng. Trong thực tế, văn bản này sẽ được dịch bởi Python backend.)`
      }
    }

    setIsLoading(false)
    return result
  }

  return { translate, isLoading }
}
