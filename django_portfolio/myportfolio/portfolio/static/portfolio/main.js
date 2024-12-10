"use strict";

const header = document.querySelector(".header");
const nav = document.querySelector(".navigation");
const navHeight = nav.getBoundingClientRect().height;
const showInfoButtons = document.querySelectorAll(".show-info");
const listInfo = document.querySelector(".list-info");
const hideInfoButtons = document.querySelectorAll(".hide-info");

const stickyNav = function (entries) {
  const [entry] = entries;
  if (!entry.isIntersecting) nav.classList.add("sticky");
  else nav.classList.remove("sticky");
};

const headerObserver = new IntersectionObserver(stickyNav, {
  root: null,
  threshold: 0,
  rootMargin: `-${navHeight}px`,
});

headerObserver.observe(header);

const displayInfo = function (e) {
  e.preventDefault();
  const btn = e.target;
  const objId = btn.dataset.proj;
  const objToDisplay = document.querySelector(`#${objId}`);
  objToDisplay.classList.remove("hidden");
};

const hideInfo = function (e) {
  e.preventDefault();
  const btn = e.target;
  const objId = btn.dataset.proj;
  const objToRemove = document.querySelector(`#${objId}`);
  objToRemove.classList.add("hidden");
};

showInfoButtons.forEach((btn) => btn.addEventListener("click", displayInfo));

hideInfoButtons.forEach((btn) => btn.addEventListener("click", hideInfo));
