$(document).ready(function () {
    // Init
    $(".image-section").hide();
    $("#downloadFile").hide()
    $(".loader").hide();
    $("#result").hide();
  
    // Upload Preview
    function readURL(input) {
      if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
          $("#imagePreview").css(
            "background-image",
            "url(" + e.target.result + ")"
          );
          $("#imagePreview").hide();
          $("#imagePreview").fadeIn(650);
        };
        reader.readAsDataURL(input.files[0]);
      }
    }
    $("#input_file").change(function () {
      $(".image-section").show();
      $("#btn-predict").show();
      $("#result").text("");
      $("#result").hide();
      $("#mainBtn").html("Uploaded");
      readURL(this);
    });
  
    // Predict
    $("#btn-predict").click(function () {
      var form_data = new FormData($("#upload-file")[0]);
  
      // Show loading animation
      $(this).hide();
      $(".loader").show();
  
      // Make prediction by calling api /predict
      $.ajax({
        type: "POST",
        url: "/predict",
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        async: true,
        success: function (data) {
          // Get and display the result
          $(".loader").hide();
          $("#result").fadeIn(600);
          $("#downloadFile").show()
          console.log("data:", data);
          $("#result").text("The Predicted number of sales is:  " + data);
          $("#mainBtn").html("Upload File");
          console.log("Success!");
        },
      });
    });
  });