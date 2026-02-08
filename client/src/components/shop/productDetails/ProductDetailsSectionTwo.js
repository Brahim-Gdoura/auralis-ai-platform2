import React, { Fragment, useContext, useEffect, useState } from "react";
import AllReviews from "./AllReviews";
import ReviewForm from "./ReviewForm";

import { ProductDetailsContext } from "./";
import { LayoutContext } from "../layout";

import { isAuthenticate } from "../auth/fetchApi";

import "./style.css";

const Menu = () => {
  const { data, dispatch } = useContext(ProductDetailsContext);
  const { data: layoutData } = useContext(LayoutContext);

  return (
    <Fragment>
      <div className="flex flex-col md:flex-row items-center justify-center border-b border-gray-200">
        <div
          onClick={(e) => dispatch({ type: "menu", payload: "description" })}
          className={`${
            data.menu === "description" ? "border-b-2 border-yellow-700" : ""
          } px-4 py-3 cursor-pointer hover:text-yellow-700 transition`}
        >
          Description
        </div>
        <div
          onClick={(e) => dispatch({ type: "menu", payload: "reviews" })}
          className={`${
            data.menu === "reviews" ? "border-b-2 border-yellow-700" : ""
          } px-4 py-3 relative flex cursor-pointer hover:text-yellow-700 transition`}
        >
          <span>Reviews</span>
          <span className="absolute text-xs top-0 right-0 mt-2 bg-yellow-700 text-white rounded px-1">
            {layoutData.singleProductDetail.pRatingsReviews.length}
          </span>
        </div>
        <div
          onClick={(e) => dispatch({ type: "menu", payload: "info" })}
          className={`${
            data.menu === "info" ? "border-b-2 border-yellow-700" : ""
          } px-4 py-3 cursor-pointer hover:text-yellow-700 transition`}
        >
          Additional Info
        </div>
      </div>
    </Fragment>
  );
};

const RatingReview = () => {
  return (
    <Fragment>
      <AllReviews />
      {isAuthenticate() ? (
        <ReviewForm />
      ) : (
        <div className="mb-12 md:mx-16 lg:mx-20 xl:mx-24 bg-red-200 px-4 py-2 rounded mb-4">
          You need to login in for review
        </div>
      )}
    </Fragment>
  );
};

const AdditionalInfo = ({ product }) => {
  return (
    <Fragment>
      <div className="space-y-4 px-4 md:px-8 py-6">
        <div className="border-b pb-4">
          <h4 className="text-gray-800 font-semibold mb-2">Features</h4>
          <ul className="text-gray-600 text-sm space-y-2">
            <li className="flex items-start">
              <span className="text-yellow-700 mr-2">✓</span>
              <span>Premium quality materials and craftsmanship</span>
            </li>
            <li className="flex items-start">
              <span className="text-yellow-700 mr-2">✓</span>
              <span>Durability tested and certified</span>
            </li>
            <li className="flex items-start">
              <span className="text-yellow-700 mr-2">✓</span>
              <span>Eco-friendly and sustainable production</span>
            </li>
            <li className="flex items-start">
              <span className="text-yellow-700 mr-2">✓</span>
              <span>Professional packaging for safe delivery</span>
            </li>
          </ul>
        </div>
        <div className="border-b pb-4">
          <h4 className="text-gray-800 font-semibold mb-2">Specifications</h4>
          <div className="text-gray-600 text-sm space-y-2">
            <div className="flex justify-between">
              <span>Product Type:</span>
              <span className="font-medium text-gray-800">
                {product.pCategory ? product.pCategory.cName : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Quality:</span>
              <span className="font-medium text-gray-800">Premium</span>
            </div>
            <div className="flex justify-between">
              <span>Warranty:</span>
              <span className="font-medium text-gray-800">1 Year</span>
            </div>
            <div className="flex justify-between">
              <span>Support:</span>
              <span className="font-medium text-gray-800">
                24/7 Customer Care
              </span>
            </div>
          </div>
        </div>
        <div>
          <h4 className="text-gray-800 font-semibold mb-2">Why Choose Us?</h4>
          <p className="text-gray-600 text-sm leading-relaxed">
            We provide only the finest quality products with exceptional
            customer service. Each item is carefully selected and tested to
            ensure it meets our high standards. Your satisfaction is our
            priority.
          </p>
        </div>
      </div>
    </Fragment>
  );
};

const ProductDetailsSectionTwo = (props) => {
  const { data } = useContext(ProductDetailsContext);
  const { data: layoutData } = useContext(LayoutContext);
  const [singleProduct, setSingleproduct] = useState({});

  useEffect(() => {
    setSingleproduct(
      layoutData.singleProductDetail ? layoutData.singleProductDetail : "",
    );

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Fragment>
      <section className="m-4 md:mx-12 md:my-8">
        <Menu />
        <div className="mt-6">
          {data.menu === "description" ? (
            <div className="text-gray-700 leading-relaxed">
              <h3 className="text-lg font-semibold mb-3">
                Product Description
              </h3>
              <p>
                {singleProduct.pDescription ||
                  "High-quality product carefully selected for you. This item combines durability, style, and functionality to meet your needs perfectly."}
              </p>
            </div>
          ) : data.menu === "reviews" ? (
            <RatingReview />
          ) : (
            <div className="mt-4 md:mx-8 lg:mx-16">
              <h3 className="text-lg font-semibold mb-4">
                Additional Information
              </h3>
              <AdditionalInfo product={singleProduct} />
            </div>
          )}
        </div>
      </section>
      <div className="m-4 md:mx-8 md:my-6 bg-gray-50 rounded-lg border border-gray-200 px-6 py-4 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider font-semibold mb-1">
              Category
            </p>
            <p className="text-gray-800">
              {singleProduct.pCategory ? singleProduct.pCategory.cName : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider font-semibold mb-1">
              Ratings
            </p>
            <p className="text-yellow-700 font-semibold">★★★★★</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider font-semibold mb-1">
              Reviews
            </p>
            <p className="text-gray-800">
              {singleProduct.pRatingsReviews
                ? singleProduct.pRatingsReviews.length
                : 0}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider font-semibold mb-1">
              Status
            </p>
            <p
              className={
                singleProduct.pQuantity > 0
                  ? "text-green-600 font-semibold"
                  : "text-red-600 font-semibold"
              }
            >
              {singleProduct.pQuantity > 0 ? "In Stock" : "Out of Stock"}
            </p>
          </div>
        </div>
      </div>
    </Fragment>
  );
};

export default ProductDetailsSectionTwo;
