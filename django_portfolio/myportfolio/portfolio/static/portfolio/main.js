


document.addEventListener('DOMContentLoaded', function () {
    const header = document.querySelector('.header');
    console.log(header)
    const nav = document.querySelector('.navigation');
    console.log(nav)
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


  });