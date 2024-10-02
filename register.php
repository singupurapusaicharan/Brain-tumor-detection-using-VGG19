This is register.php

<?php
session_start();
if (isset($_SESSION["register"])) {
    header("Location: index.php");
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Register Form | Brain Tumor Detection</title>
  <link rel="stylesheet" href="login.css">
  <link href='https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css' rel='stylesheet'>
</head>
<body>

  <div class="logo">
    <i class='bx bxs-brain'></i>
    Brain Tumor Detection
  </div>
  
  <div class="wrapper">    
    <?php
        if (isset($_POST["submit"])) {
           $name = $_POST["name"];
           $email = $_POST["email"];
           $password = $_POST["password"];
           $passwordHash = password_hash($password, PASSWORD_DEFAULT);

           $errors = array();
           if (empty($name) OR empty($email) OR empty($password)) {
               array_push($errors, "All fields are required");
           }
           if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
               array_push($errors, "Email is not valid");
           }
           if (strlen($password) < 6) {
               array_push($errors, "Password must be at least 6 characters long");
           }
           require_once "server.php";
           $sql = "SELECT * FROM register WHERE email = '$email'";
           $result = mysqli_query($conn, $sql);
           $rowCount = mysqli_num_rows($result);
           if ($rowCount > 0) {
               array_push($errors, "Email already exists!");
           }
           if (count($errors) > 0) {
               foreach ($errors as $error) {
                   echo "<div class='alert alert-danger'>$error</div>";
               }
           } else {


               $sql = "INSERT INTO register (name, email, password) VALUES (?, ?, ?)";
               $stmt = mysqli_stmt_init($conn);
               $prepareStmt = mysqli_stmt_prepare($stmt, $sql);
               if ($prepareStmt) {
                   mysqli_stmt_bind_param($stmt, "sss", $name, $email, $passwordHash);
                   mysqli_stmt_execute($stmt);
                   header("location: index.php");
                   
               } else {
                   alert("Something went wrong");
               }
           }
        }
    ?>

    <form action="register.php" method="post">
      <h1>Register</h1>
      
      <div class="input-box">
        <input type="text" name="name" placeholder="Name" required>
        <i class='bx bxs-user'></i>
      </div>
      <div class="input-box">
        <input type="email" name="email" placeholder="Email" required>
        <i class='bx bxs-envelope'></i>
      </div>
      <div class="input-box">
        <input type="password" name="password" placeholder="Password" required>
        <i class='bx bxs-lock-alt'></i>
      </div>
      <button type="submit" name="submit" class="btn">Register</button>
    </form>
    <div>
        <p>Already have an account? <a href="login.php">Login</a></p>
    </div>
  </div>
</body>
</html>