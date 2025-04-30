// src/services/suggestionService.ts
import { Suggestion } from '@/components/CodeEditor';

export class SuggestionService {
  /* ------------------------------------------------------------------ *
   *  Local pattern-based suggestions (unchanged from your original)    *
   * ------------------------------------------------------------------ */
  static getSuggestions(
    code: string,
    cursorPosition: { line: number; ch: number }
  ): Suggestion[] {
    const suggestions: Suggestion[] = [];

    if (code.includes('def ') && !code.includes('return')) {
      suggestions.push({
        text: 'return result',
        description: 'Return a value from your function',
      });
    }

    if (code.includes('print') || code.includes('input')) {
      suggestions.push({
        text: 'input("Enter value: ")',
        description: 'Get user input',
      });
    }

    if (code.toLowerCase().includes('list') || code.includes('[]')) {
      suggestions.push({
        text: 'for item in my_list:\n    print(item)',
        description: 'Iterate through a list',
      });
    }

    // always provide a couple of generic snippets
    suggestions.push(
      {
        text: 'if condition:\n    # do something',
        description: 'Conditional statement',
      },
      {
        text: 'try:\n    # code\nexcept Exception as e:\n    print(e)',
        description: 'Exception handling',
      }
    );

    return suggestions;
  }

  /* ------------------------------------------------------------------ *
   *  NEW: Call the local inference service at http://127.0.0.1:5000    *
   * ------------------------------------------------------------------ */
  static async generateCode(
    prompt: string
  ): Promise<{ generated_text: string }> {
    const res = await fetch('http://127.0.0.1:5000/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    if (!res.ok) {
      throw new Error(`Inference API error: ${res.status} ${res.statusText}`);
    }

    return res.json();
  }
}
