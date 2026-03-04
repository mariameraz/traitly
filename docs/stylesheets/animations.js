const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.animate').forEach(el => {
  const rect = el.getBoundingClientRect();
  if (rect.top < window.innerHeight && rect.bottom > 0) {
    el.classList.add('visible'); 
  } else {
    observer.observe(el); 
  }
});