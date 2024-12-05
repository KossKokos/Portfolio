
const test_h1 = document.querySelector('h1');
console.log(test_h1)


// document.addEventListener('DOMContentLoaded', function () {
    const header = document.querySelector('.header');
    const nav = document.querySelector('.navigation');
    const navHeight = nav.getBoundingClientRect().height;


    const stickyNav = function (entries) {
      const [entry] = entries;
      if (!entry.isIntersecting) nav.classList.add('sticky');
      else nav.classList.remove('sticky');
    };
  
    const headerObserver = new IntersectionObserver(stickyNav, {
      root: null,
      threshold: 0,
      rootMargin: `-${navHeight}px`,
    });
  
    headerObserver.observe(header);
  // });

    if (test_h1) {
      test_h1.style.marginTop = `${navHeight}px`;
    };
  