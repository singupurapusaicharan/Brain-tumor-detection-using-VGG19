document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission
  
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
  
    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();
  
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(username)) {
      alert('Please enter a valid email address.');
      usernameInput.focus();
      return;
    }
  
    // Validate password format
    const passwordRegex = /^(?=.[a-zA-Z])(?=.\d)(?=.[@$!%?&])[A-Za-z\d@$!%?&]{6,}$/;
    if (!passwordRegex.test(password)) {
      alert('Password must be at least 6 characters long and contain at least one letter, one number, and one special character.');
      passwordInput.focus();
      return;
    }
  
    // Redirect to index.html
    window.location.assign("index.html");
  });