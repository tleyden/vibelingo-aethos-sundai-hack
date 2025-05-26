document.addEventListener('DOMContentLoaded', () => {
    // Check if we're loaded from /static/ path and adjust base URL if needed
    const baseUrl = window.location.pathname.includes('/static/') ? '..' : '';
    
    // Track if we're in drill mode
    let inDrillMode = false;
    let currentWordPair = null;
    let currentDirection = null;

    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const startButton = document.getElementById('start-btn');
    const progressBar = document.getElementById('progress-bar');
    
    // Generate a random session ID for this chat session
    const sessionId = Math.random().toString(36).substring(2, 15);
    let chatCompleted = false;
    
    // Function to add a message to the chat
    function addMessage(text, isUser = false, isDrill = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        if (isDrill && !isUser) {
            messageDiv.classList.add('drill');
        }
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to add a drill message with feedback
    function addDrillMessage(question, evaluation = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot drill';
        
        // Add evaluation feedback if available
        if (evaluation) {
            const feedbackPara = document.createElement('p');
            feedbackPara.className = evaluation.correct ? 'feedback correct' : 'feedback incorrect';
            feedbackPara.textContent = evaluation.feedback;
            
            if (evaluation.explanation) {
                const explanationPara = document.createElement('p');
                explanationPara.className = 'explanation';
                explanationPara.textContent = evaluation.explanation;
                feedbackPara.appendChild(explanationPara);
            }
            
            messageDiv.appendChild(feedbackPara);
        }
        
        // Create a container for the question content
        const questionContainer = document.createElement('div');
        questionContainer.className = 'question-container';
        
        // Extract image URL if present
        const imgRegex = /!\[.*?\]\((.*?)\)/;
        const imgMatch = question.match(imgRegex);
        
        if (imgMatch && imgMatch[1]) {
            // We found an image, extract the URL
            const imageUrl = imgMatch[1];
            
            // Add a title at the top
            const titlePara = document.createElement('p');
            titlePara.textContent = "Describe this image in German:";
            questionContainer.appendChild(titlePara);
            
            // Create and add the image
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = 'Drill Image';
            img.className = 'drill-image';
            questionContainer.appendChild(img);
            
            // Process the rest of the content
            // Remove the image markdown and split the remaining text
            const remainingText = question.replace(imgRegex, '').trim();
            const sections = remainingText.split(/\*\*([^*]+)\*\*/);
            
            let currentSection = '';
            let isHeader = false;
            
            // Process each section
            for (let i = 0; i < sections.length; i++) {
                if (sections[i].trim() === '') continue;
                
                if (i % 2 === 1) { // This is a header (was between ** **)
                    // Skip the image prompt section if present
                    if (sections[i].trim() === 'Image prompt:') {
                        // Skip this section and the next one (which contains the prompt text)
                        i++; // Skip the next section
                        continue;
                    }
                    
                    // If we have accumulated text, add it first
                    if (currentSection.trim()) {
                        const para = document.createElement('p');
                        para.textContent = currentSection.trim();
                        questionContainer.appendChild(para);
                        currentSection = '';
                    }
                    
                    // Add the header
                    const header = document.createElement('h3');
                    header.textContent = sections[i];
                    header.className = 'section-header';
                    questionContainer.appendChild(header);
                    isHeader = true;
                } else {
                    // This is regular text or a list
                    if (isHeader && sections[i].includes('•')) {
                        // This is a list that follows a header
                        const listContainer = document.createElement('ul');
                        listContainer.className = 'vocab-list';
                        
                        // Split by bullet points
                        const items = sections[i].split('•').filter(item => item.trim());
                        
                        items.forEach(item => {
                            const listItem = document.createElement('li');
                            listItem.textContent = item.trim();
                            listContainer.appendChild(listItem);
                        });
                        
                        questionContainer.appendChild(listContainer);
                    } else {
                        // Regular text, accumulate it
                        currentSection += sections[i];
                    }
                    
                    isHeader = false;
                }
            }
            
            // Add any remaining text
            if (currentSection.trim()) {
                const para = document.createElement('p');
                para.textContent = currentSection.trim();
                questionContainer.appendChild(para);
            }
        } else {
            // No image, just add the text as is
            const textPara = document.createElement('p');
            textPara.textContent = question;
            questionContainer.appendChild(textPara);
        }
        
        messageDiv.appendChild(questionContainer);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send a message to the API
    async function sendMessage(message) {
        try {
            const response = await fetch(`${baseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Update progress bar
            const progressPercentage = (data.progress / 5) * 100;
            progressBar.style.width = `${progressPercentage}%`;
            
            // Check if we're in drill mode
            if (data.is_drill) {
                inDrillMode = true;
                currentWordPair = data.word_pair;
                currentDirection = data.direction;
                
                // Add drill message
                addDrillMessage(data.question, data.evaluation);
            } else {
                // Add regular bot message
                addMessage(data.question);
            }
            
            // Check if chat is completed
            if (data.completed && !inDrillMode) {
                // This is the end of onboarding, we'll transition to drill mode
                addMessage('Great! Now let\'s start practicing some vocabulary!');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your request. Please try again.');
        }
    }
    
    // Function to fetch session data
    async function fetchSessionData() {
        try {
            const response = await fetch(`${baseUrl}/session/${sessionId}`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Display the collected answers
            addMessage('Here are your onboarding answers:');
            data.onboarding_answers.forEach((answer, index) => {
                addMessage(`Answer ${index + 1}: ${answer}`);
            });
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error fetching your session data.');
        }
    }
    
    // Event listener for send button
    sendButton.addEventListener('click', () => {
        const message = userInput.value.trim();
        
        if (message && !chatCompleted) {
            addMessage(message, true);
            userInput.value = '';
            sendMessage(message);
        }
    });
    
    // Event listener for input field (send on Enter key)
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });
    
    // Event listener for start button
    startButton.addEventListener('click', () => {
        startButton.style.display = 'none';
        userInput.disabled = false;
        sendButton.disabled = false;
        
        // Start the chat
        sendMessage('');
    });
});
