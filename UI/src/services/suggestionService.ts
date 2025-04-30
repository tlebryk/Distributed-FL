import React, { useState, useEffect } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import SuggestionCard from './SuggestionCard';

export interface Suggestion {
  text: string;
  description?: string;
}

interface CodeEditorProps {
  initialCode?: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  initialCode = '# Write your Python code here\ndef hello_world():\n    print("Hello, World!")\n\nhello_world()',
}) => {
  const [code, setCode] = useState<string>(initialCode);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([
    { text: 'print("Hello, World!")', description: 'Print a greeting message' },
    { text: 'def function_name(param):', description: 'Define a new function' },
    { text: 'for item in items:', description: 'Loop through items' },
  ]);
  const [cursorPosition, setCursorPosition] = useState({ line: 0, ch: 0 });
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState<number>(-1);

  // New state for AI generation
  const [generatedText, setGeneratedText] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      // TODO: replace with federated learning AI service
      const newSuggestions = []; // keep placeholder or remove
      setSuggestions(newSuggestions);
    }, 500);

    return () => clearTimeout(timer);
  }, [code, cursorPosition]);

  const handleChange = React.useCallback((value: string) => {
    setCode(value);
  }, []);

  const insertSuggestion = (suggestion: Suggestion) => {
    setCode(code + '\n' + suggestion.text);
    setActiveSuggestionIndex(-1);
  };

  // Trigger your inference endpoint
  const generateFromAI = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: code }),
      });
      const data = await response.json();
      setGeneratedText(data.generated_text || '');
    } catch (err) {
      console.error('Generation error:', err);
      setGeneratedText('Error generating code');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestionIndex(prev => prev < suggestions.length - 1 ? prev + 1 : 0);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestionIndex(prev => prev > 0 ? prev - 1 : suggestions.length - 1);
    } else if (e.key === 'Enter' && activeSuggestionIndex >= 0) {
      e.preventDefault();
      insertSuggestion(suggestions[activeSuggestionIndex]);
    } else if (e.key === 'Escape') {
      setActiveSuggestionIndex(-1);
    }
  };

  return (
    <div className= "flex flex-col md:flex-row gap-4 w-full h-full" >
    <Card className="flex-1" >
      <CardHeader>
      <CardTitle>Python Code Editor </CardTitle>
        </CardHeader>
        < CardContent >
        <div className="border rounded-md overflow-hidden" >
          <div onKeyDown={ handleKeyDown } tabIndex = { 0} className = "focus:outline-none" >
            <CodeMirror
                value={ code }
  height = "400px"
  extensions = { [python()]}
  onChange = { handleChange }
  theme = "dark"
  className = "text-sm"
  basicSetup = {{
    lineNumbers: true,
      highlightActiveLine: true,
                }
}
onUpdate = { update => {
  if (update.docChanged) {
    setCursorPosition({ line: 0, ch: 0 });
  }
}}
              />
  </div>
  </div>

{/* AI Generate Button */ }
<div className="mt-4 flex items-center space-x-2" >
  <button
              onClick={ generateFromAI }
disabled = { isGenerating }
className = "px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
  >
  { isGenerating? 'Generating...': 'Generate Code' }
  </button>
  </div>

{/* Generated response */ }
{
  generatedText && (
    <div className="mt-4 p-4 bg-gray-800 text-white rounded whitespace-pre-wrap" >
      { generatedText }
      </div>
          )
}
</CardContent>
  </Card>

  < Card className = "w-full md:w-64" >
    <CardHeader>
    <CardTitle>Suggestions </CardTitle>
    </CardHeader>
    < CardContent >
    <div className="space-y-2" >
    {
      suggestions.map((suggestion, index) => (
        <SuggestionCard
                key= { index }
                suggestion = { suggestion }
                onClick = {() => insertSuggestion(suggestion)}
isActive = { index === activeSuggestionIndex}
              />
            ))}
<Separator className="my-2" />
  <div className="p-2 bg-blue-50 border border-blue-100 rounded-md text-xs text-blue-500 italic" >
    <div className="font-semibold" > AI Integration Placeholder </div>
      < div className = "mt-1" > Static examples above, will edit these suggestins to use our federated learning model </div>
        </div>
        </div>
        </CardContent>
        </Card>
        </div>
  );
};

export default CodeEditor;
