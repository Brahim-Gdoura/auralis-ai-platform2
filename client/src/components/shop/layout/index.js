import React, { Fragment, createContext } from "react";
import { Navber, Footer, CartModal } from "../partials";
import LoginSignup from "../auth/LoginSignup";
import Chatbot from "../home/Chatbot";

export const LayoutContext = createContext();

const Layout = ({ children }) => {
  return (
    <Fragment>
      <div className="flex-grow">
        <Navber />
        <LoginSignup />
        <CartModal />
        {children}
      </div>
      <Footer />
      <Chatbot />
    </Fragment>
  );
};

export default Layout;