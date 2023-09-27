document.addEventListener("DOMContentLoaded", function () {
    const text = document.getElementById('text');
    const leaf = document.getElementById('leaf');
    const hill1 = document.getElementById('hill1');
    const hill4 = document.getElementById('hill4');
    const hill5 = document.getElementById('hill5');
    const menuToggle = document.getElementById("menu-toggle");
    const navLinks = document.querySelector(".nav-links");
    const fileInput = document.getElementById("fileInput");
    const uploadMessage = document.getElementById("upload-message");

    window.addEventListener('scroll', () => {
        let value = window.scrollY;
        text.style.marginTop = value * 2.5 + 'px';
        leaf.style.top = value * -1.5 + 'px';
        leaf.style.left = value * 1.5 + 'px';
        hill5.style.left = value * 1.5 + 'px';
        hill4.style.left = value * -1.5 + 'px';
        hill1.style.top = value * 1 + 'px';
    });

    menuToggle.addEventListener("click", function () {
        navLinks.classList.toggle("active");
    });

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            // Simulate an upload process (you can replace this with your actual upload code)
            uploadMessage.textContent = "Uploading...";
            setTimeout(function () {
                uploadMessage.textContent = "Successfully Uploaded!";
                // Reset the message after 2 seconds
                setTimeout(function () {
                    uploadMessage.textContent = "";
                }, 2000);
            }, 2000); // Display the message for 2 seconds (you can adjust the duration)
        }
    });

    // Function to highlight the clicked navigation button
    function highlightButton(buttonId) {
        const navLinks = document.querySelectorAll(".nav a");
    
        navLinks.forEach(function (link) {
            link.classList.remove("active");
        });
    
        const selectedLink = document.querySelector(`.nav a[href$="${buttonId}.html"]`);
        if (selectedLink) {
            selectedLink.classList.add("active");
        }
    }

    // Add click event listeners to the navigation links
    const navLinksList = document.querySelectorAll(".nav a");
    navLinksList.forEach(function (link) {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            const href = this.getAttribute("href");
            const buttonId = href.split("/").pop().split(".")[0];
            highlightButton(buttonId);
        });
    });
});
function toggleDarkMode() {
    const body = document.querySelector('body');
    body.classList.toggle('dark-mode');
}
const toggleButton = document.getElementById('toggle-mode');
toggleButton.addEventListener('click', toggleDarkMode);



