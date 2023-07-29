<script>
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('jobApplicationForm').addEventListener('submit', function(e) {
        e.preventDefault();
        alert('Thank you for the application!');
        this.reset();
    });
});
</script>
