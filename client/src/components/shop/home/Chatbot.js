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
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      type: "user",
      text: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          type: "bot",
          text: data.response,
          sources: data.sources,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error(data.error || "Failed to get response");
      }
    } catch (error) {
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
        style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 9999 }}
        className="w-16 h-16 rounded-full shadow-[var(--shadow-xl)] flex items-center justify-center transition-all duration-300 hover:scale-110 bg-gradient-to-br from-primary via-secondary to-accent animate-pulse-glow bg-white"
      >
        <MessageCircle className="w-8 h-8 text-primary-foreground" strokeWidth={2} />
        <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-background animate-pulse"></div>
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
      className="bg-white rounded-2xl shadow-[var(--shadow-xl)] flex flex-col overflow-hidden animate-scale-in"
    >
      <div className="bg-gradient-to-r from-blue-200 via-blue-300 to-accent px-5 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <img 
              src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
              alt="Auralis" 
              className="w-10 h-10 rounded-full border-2 border-primary-foreground shadow-[var(--shadow-md)] bg-background"
            />
            <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-primary-foreground"></div>
          </div>
          <div>
            <h3 className="text-primary-foreground font-semibold text-base">Auralis</h3>
            <p className="text-primary-foreground/90 text-xs flex items-center">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full mr-1.5 animate-pulse"></span>
              Online
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-primary-foreground/90 hover:bg-white/20 rounded-full p-1.5 transition-all duration-200"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div 
        className="flex-1 overflow-y-auto px-4 py-3 space-y-3"
        style={{ 
          maxHeight: 'calc(600px - 140px)',
          background: 'var(--gradient-chat)'
        }}
      >
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.type === "user" ? "justify-end " : "justify-start "} animate-slide-up`}
          >
            <div className={`flex items-end space-x-2 max-w-[80%] ${message.type === "user" ? "flex-row-reverse  space-x-reverse" : ""}`}>
              {message.type === "bot" && (
                <img 
                  src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
                  alt="Auralis" 
                  className="w-6 h-6 rounded-full flex-shrink-0 mb-1"
                />
              )}
              <div>
                <div
                  className={`px-4 py-2.5 shadow-[var(--shadow-sm)] transition-all duration-200 hover:shadow-[var(--shadow-md)] ${
                    message.type === "user"
                      ? "bg-blue-100 text-primary-foreground rounded-t-2xl rounded-l-2xl rounded-br-md"
                      : "bg-green-100 text-card-foreground rounded-t-2xl rounded-r-2xl rounded-bl-md border border-border"
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                </div>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-1.5 px-2.5 py-1 bg-primary/10 rounded-lg text-xs text-primary inline-block">
                    📚 {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
                  </div>
                )}
                <div className={`flex items-center space-x-1 mt-1 px-1 ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                  <p className="text-xs text-muted-foreground">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                  {message.type === "user" && (
                    <svg className="w-4 h-4 text-primary" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/>
                    </svg>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start animate-fade-in">
            <div className="flex items-end space-x-2">
              <img 
                src="https://api.dicebear.com/7.x/bottts/svg?seed=Auralis&backgroundColor=ffffff" 
                alt="Auralis" 
                className="w-6 h-6 rounded-full flex-shrink-0 mb-1"
              />
              <div className="bg-card rounded-t-2xl rounded-r-2xl rounded-bl-md px-4 py-3 shadow-[var(--shadow-sm)] border border-border">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="px-4 py-3 bg-card border-t border-border">
        <div className="flex items-center space-x-2 bg-muted rounded-full px-4 py-2.5 border border-border focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all duration-200">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isLoading}
            className="flex-1 bg-transparent focus:outline-none text-sm disabled:cursor-not-allowed placeholder-muted-foreground text-foreground"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-gradient-to-br from-primary to-secondary text-primary-foreground rounded-full p-2 hover:scale-105 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-[var(--shadow-md)]"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
