import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send } from "lucide-react";

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: "bot",
      text: "Hey! 👋 I'm Auralis, your shopping assistant. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const messagesEndRef = useRef(null);

  // Initialize session ID on component mount
  useEffect(() => {
    let id = localStorage.getItem("session_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("session_id", id);
    }
    setSessionId(id);
  }, []);

  // Load chat history on mount
  useEffect(() => {
    if (sessionId) {
      loadHistory();
    }
  }, [sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load previous chat history from backend
  const loadHistory = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await response.json();
      if (data.history && data.history.length > 0) {
        const historyMessages = data.history.map(msg => ({
          type: msg.role,
          text: msg.content,
          timestamp: new Date(msg.timestamp || Date.now()),
          sources: msg.sources || []
        }));
        // Replace initial greeting with actual history
        setMessages(historyMessages.length > 0 ? historyMessages : messages);
      }
    } catch (err) {
      console.error("Error loading history:", err);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      type: "user",
      text: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageContent = input;
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          session_id: sessionId,
          query: messageContent 
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          type: "bot",
          text: data.answer || data.response,
          sources: data.sources || [],
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error(data.error || "Failed to get response");
      }
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = {
        type: "bot",
        text: "Oops! Something went wrong. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        style={{ 
          position: 'fixed', 
          bottom: '24px', 
          right: '24px', 
          zIndex: 9999,
          width: '64px',
          height: '64px',
        }}
        className="rounded-full shadow-lg flex items-center justify-center transition-all duration-300 hover:scale-110 bg-gradient-to-br from-blue-500 via-blue-400 to-blue-600 animate-pulse"
      >
        <MessageCircle className="w-8 h-8 text-white" strokeWidth={2} />
        <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse"></div>
      </button>
    );
  }

  return (
    <div
      style={{ 
        position: 'fixed', 
        bottom: '24px', 
        right: '24px', 
        width: '380px', 
        height: '600px',
        zIndex: 9999 
      }}
      className="bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-400 via-blue-500 to-blue-600 px-5 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <img 
              src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
              alt="Auralis" 
              className="w-10 h-10 rounded-full border-2 border-white shadow-md bg-white"
            />
            <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
          </div>
          <div>
            <h3 className="text-white font-semibold text-base">Auralis</h3>
            <p className="text-blue-100 text-xs flex items-center">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full mr-1.5 animate-pulse"></span>
              Online
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-white/90 hover:bg-white/20 rounded-full p-1.5 transition-all duration-200"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages Container */}
      <div 
        className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-gray-50"
        style={{ 
          maxHeight: 'calc(600px - 140px)',
        }}
      >
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
          >
            <div className={`flex items-end space-x-2 max-w-[80%] ${message.type === "user" ? "flex-row-reverse space-x-reverse" : ""}`}>
              {message.type === "bot" && (
                <img 
                  src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
                  alt="Auralis" 
                  className="w-6 h-6 rounded-full flex-shrink-0 mb-1"
                />
              )}
              <div>
                <div
                  className={`px-4 py-2.5 shadow-sm transition-all duration-200 hover:shadow-md ${
                    message.type === "user"
                      ? "bg-blue-500 text-white rounded-t-2xl rounded-l-2xl rounded-br-md"
                      : "bg-white text-gray-800 rounded-t-2xl rounded-r-2xl rounded-bl-md border border-gray-200"
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                </div>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-1.5 px-2.5 py-1 bg-blue-50 rounded-lg text-xs text-blue-600 inline-block">
                    📚 {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
                  </div>
                )}
                <div className={`flex items-center space-x-1 mt-1 px-1 ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                  <p className="text-xs text-gray-500">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                  {message.type === "user" && (
                    <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/>
                    </svg>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-end space-x-2">
              <img 
                src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
                alt="Auralis" 
                className="w-6 h-6 rounded-full flex-shrink-0 mb-1"
              />
              <div className="bg-white rounded-t-2xl rounded-r-2xl rounded-bl-md px-4 py-3 shadow-sm border border-gray-200">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="px-4 py-3 bg-white border-t border-gray-200">
        <div className="flex items-center space-x-2 bg-gray-100 rounded-full px-4 py-2.5 border border-gray-300 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-200 transition-all duration-200">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isLoading}
            className="flex-1 bg-transparent focus:outline-none text-sm disabled:cursor-not-allowed placeholder-gray-400 text-gray-800"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-full p-2 hover:scale-105 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-md"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;