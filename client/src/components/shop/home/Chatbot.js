import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, ShoppingBag, Star } from "lucide-react";
import './Chatbot.css';
import auralisAvatar from "./Auralis_bot.png";

// Product Card Component
const ProductCard = ({ product }) => {
  return (
    <div className="product-card">
      <div className="product-image-container">
        <img 
          src={product.image || "https://via.placeholder.com/300x200?text=Product+Image"} 
          alt={product.name}
          className="product-image"
        />
        {product.stock && product.stock < 10 && (
          <div className="product-badge product-badge-limited">
            Only {product.stock} left!
          </div>
        )}
        {product.badge && (
          <div className="product-badge product-badge-featured">
            <Star className="badge-icon" />
            {product.badge}
          </div>
        )}
      </div>
      
      <div className="product-content">
        <h4 className="product-name">{product.name}</h4>
        {product.category && (
          <p className="product-category">{product.category}</p>
        )}
        
        {product.description && (
          <p className="product-description">{product.description}</p>
        )}
        
        <div className="product-footer">
          <div className="product-price-container">
            <span className="product-price">${product.price}</span>
            {product.originalPrice && (
              <span className="product-original-price">${product.originalPrice}</span>
            )}
          </div>
          
          {product.rating && (
            <div className="product-rating">
              <Star className="rating-icon" />
              <span className="rating-text">{product.rating}</span>
            </div>
          )}
        </div>
        
        {product.stock !== undefined && (
          <div className="product-stock">
            <span className={product.stock > 0 ? 'stock-available' : 'stock-unavailable'}>
              {product.stock > 0 ? `✓ ${product.stock} in stock` : '✗ Out of stock'}
            </span>
          </div>
        )}
        
        <button className="product-add-to-cart">
          <ShoppingBag className="cart-icon" />
          Add to Cart
        </button>
      </div>
    </div>
  );
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: "bot",
      text: "Hey there! 👋 I'm Auralis, your personal shopping companion. I'm here to help you discover amazing products tailored just for you! What are you looking for today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    let storedSessionId = localStorage.getItem("chatbot_session_id");
    if (!storedSessionId) {
      storedSessionId = crypto.randomUUID();
      localStorage.setItem("chatbot_session_id", storedSessionId);
    }
    setSessionId(storedSessionId);
    loadChatHistory(storedSessionId);
  }, []);

  const loadChatHistory = async (sessionId) => {
    try {
      const response = await fetch("http://localhost:5000/api/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      const data = await response.json();
      if (data.history && data.history.length > 0) {
        const formattedMessages = data.history.map(msg => ({
          type: msg.role === "user" ? "user" : "bot",
          text: msg.content,
          timestamp: new Date(),
        }));
        setMessages(prev => [...prev, ...formattedMessages]);
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading || !sessionId) return;

    const userMessage = {
      type: "user",
      text: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageToSend = input;
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: messageToSend,
          session_id: sessionId 
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          type: "bot",
          text: data.response || data.answer,
          sources: data.sources,
          products: data.products || [],
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error(data.error || "Failed to get response");
      }
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage = {
        type: "bot",
        text: "Oops! Something went wrong on my end. Let me try again! 🔄",
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
      <button onClick={() => setIsOpen(true)} className="chatbot-toggle">
        <MessageCircle className="toggle-icon" />
        <div className="toggle-status-indicator"></div>
      </button>
    );
  }

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="header-bg-shape header-bg-shape-1"></div>
        <div className="header-bg-shape header-bg-shape-2"></div>
        
        <div className="header-content">
          <div className="header-left">
            <div className="avatar-container">
              <div className="avatar-glow"></div>
              <img 
                src={auralisAvatar}
                alt="Auralis" 
                className="avatar-image"
              />
              <div className="avatar-status"></div>
            </div>
            <div className="header-info">
              <h3 className="header-title">
                Auralis
                <span className="header-sparkle">✨</span>
              </h3>
              <p className="header-status">
                <span className="status-dot"></span>
                Always here to help
              </p>
            </div>
          </div>
          <button onClick={() => setIsOpen(false)} className="close-button">
            <X className="close-icon" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index}>
            <div className={`message-wrapper ${message.type === "user" ? "message-user" : "message-bot"}`}>
              <div className="message-content-wrapper">
                {message.type === "bot" && (
                  <img 
                    src={auralisAvatar} 
                    alt="Auralis" 
                    className="message-avatar"
                  />
                )}
                <div>
                  <div className={`message-bubble ${message.type === "user" ? "bubble-user" : "bubble-bot"}`}>
                    <p className="message-text">{message.text}</p>
                  </div>
                  
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      📚 {message.sources.length} source{message.sources.length > 1 ? 's' : ''} referenced
                    </div>
                  )}
                  
                  <div className={`message-timestamp ${message.type === "user" ? "timestamp-user" : "timestamp-bot"}`}>
                    <p className="timestamp-text">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </p>
                    {message.type === "user" && (
                      <svg className="checkmark" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/>
                      </svg>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Product Cards */}
            {message.products && message.products.length > 0 && (
              <div className="products-container">
                {message.products.map((product, idx) => (
                  <ProductCard key={idx} product={product} />
                ))}
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message-wrapper message-bot">
            <div className="message-content-wrapper">
              <img 
                src={auralisAvatar} 
                alt="Auralis" 
                className="message-avatar"
              />
              <div className="message-bubble bubble-bot">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot" style={{ animationDelay: '0.2s' }}></div>
                  <div className="typing-dot" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="input-container">
        <div className="input-wrapper">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            disabled={isLoading}
            className="input-field"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            <Send className="send-icon" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;