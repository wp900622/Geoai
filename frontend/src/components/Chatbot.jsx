import { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send } from 'lucide-react';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your GeoAI Traffic Assistant. How can I help you understand the current or predicted traffic conditions in Taipei?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content, history: messages })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I cannot connect to the backend AI API at localhost:8000/chat right now.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`${isOpen ? 'hidden' : 'flex'} fixed bottom-6 right-6 h-14 w-14 bg-blue-600 hover:bg-blue-500 text-white rounded-full items-center justify-center shadow-[0_0_20px_rgba(37,99,235,0.4)] transition-transform hover:scale-110 z-[100]`}
      >
        <MessageCircle size={28} />
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-[350px] h-[500px] bg-slate-900 border border-slate-700 shadow-2xl rounded-2xl flex flex-col z-[100] overflow-hidden transform transition-all ring-1 ring-white/10">
          {/* Header */}
          <div className="bg-slate-800/80 backdrop-blur-md p-4 border-b border-slate-700 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 bg-blue-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.8)]"></div>
              <h3 className="text-blue-400 font-semibold tracking-wide">GeoAI Assistant</h3>
            </div>
            <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white transition-colors bg-slate-800 p-1 rounded-full hover:bg-slate-700">
              <X size={18} />
            </button>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 font-sans max-h-full">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl p-3 text-[14px] leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-sm shadow-md' 
                    : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-tl-sm shadow-sm'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-4 text-sm flex gap-1.5 items-center rounded-tl-sm shadow-sm">
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <form onSubmit={handleSend} className="p-3 border-t border-slate-700 bg-slate-800/80 backdrop-blur-md">
            <div className="relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about traffic conditions..."
                className="w-full bg-slate-900/50 border border-slate-600 text-sm text-white rounded-full pl-4 pr-12 py-3 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-inner"
                disabled={isLoading}
              />
              <button 
                type="submit" 
                disabled={isLoading || !input.trim()}
                className="absolute right-2 p-2 bg-blue-600/20 text-blue-400 hover:bg-blue-600 hover:text-white rounded-full transition-colors disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-blue-500"
              >
                <Send size={18} />
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
};

export default Chatbot;
