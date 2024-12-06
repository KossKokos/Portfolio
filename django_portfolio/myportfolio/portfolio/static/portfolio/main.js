const test_h1 = document.querySelector("h1");

// document.addEventListener('DOMContentLoaded', function () {
const header = document.querySelector(".header");
const nav = document.querySelector(".navigation");
const navHeight = nav.getBoundingClientRect().height;
// const showImageBtns = document.querySelectorAll('.show-image');


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
// });

if (test_h1) {
  const test1Marging = 0.0625 * navHeight;
  test_h1.style.marginTop = `${test1Marging}rem`;
}

// const openModal = function (e) {
//   e.preventDefault();
//   modal.classList.remove('hidden');
//   overlay.classList.remove('hidden');
// };

// const closeModal = function () {
//   modal.classList.add('hidden');
//   overlay.classList.add('hidden');
// };

// btnsOpenModal.forEach(function (btn) {
//   btn.addEventListener('click', openModal);
// });

// btnCloseModal.addEventListener('click', closeModal);

// showImageBtns.addEventListener()