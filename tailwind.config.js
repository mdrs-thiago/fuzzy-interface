/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./ui/index.html", "./ui/app.jsx"],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                puc: {
                    red: '#8C3A22',
                    stone: '#F4F5F0',
                    gray: '#666666',
                },
                zinc: {
                    850: '#27272a',
                    950: '#09090b',
                }
            }
        }
    }
};
