<script>
    document.getElementById('jobApplicationForm').addEventListener('submit', function(e) {
        e.preventDefault(); // prevent the form from submitting
        
        // Now you can send form data to your server using AJAX here if needed.
        
        // Display thank you message
        alert('Thank you for the application!');
        
        // Optionally, clear the form fields
        this.reset();
    });
</script>
