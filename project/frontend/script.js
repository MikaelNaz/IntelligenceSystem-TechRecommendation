let userId = localStorage.getItem("userId");

if (!userId) {
    userId = "user_" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("userId", userId);
}
let chatMessages = JSON.parse(localStorage.getItem("chatMessages")) || [];

function sendMessage() {
    let userInput = document.getElementById("userInput").value.trim();
    console.log("Пользователь ввел:", userInput);
    
    if (userInput !== "") {
        addMessageToChat({ type: "user", text: userInput }, true);
        saveChatHistory();
        
        fetch("http://127.0.0.1:8000/chat/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput, user_id: userId })
        })
        .then(response => {
            console.log("Статус ответа:", response.status);
            if (!response.ok) {
                throw new Error('Сетевой ответ не был успешным');
            }
            return response.json();
        })
        .then(data => {
            console.log("Данные от сервера:", data);
            addMessageToChat({ type: "system", text: data.response }, true);
            saveChatHistory();
            
            if (data.response.includes("Анализируем")) {
                getRecommendation();
            }
            loadLatestRecommendation();
        })
        .catch(error => {
            console.error("Ошибка запроса:", error);
            addMessageToChat({
                type: "system",
                text: "Ошибка соединения с сервером. Проверьте, что backend запущен."
            }, true);
        });

        document.getElementById("userInput").value = "";
    }
}

function addMessageToChat(message, save = false) {
    console.log("Добавляем сообщение:", message);
    if (save) {
        chatMessages.push(message);
    }
    
    let chatBox = document.getElementById("chat");
    let messageElement = document.createElement("div");
    messageElement.className = `chat-message ${message.type}`;
    
    let textElement = document.createElement("p");
    textElement.textContent = message.text;
    messageElement.appendChild(textElement);
    
    chatBox.appendChild(messageElement);
    
    setTimeout(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 100);
}

function getRecommendation() {
    console.log("Запуск getRecommendation");
    fetch(`http://127.0.0.1:8000/projects/${userId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка загрузки рекомендаций');
            }
            return response.json();
        })
        .then(data => {
            let recommendationsDiv = document.getElementById("recommendations");
            // Вызов функции после получения ответа от бота
            console.log("Обновляем рекомендации:", data);
        })
        .catch(error => {
            console.error("Ошибка при получении рекомендаций:", error);
            document.getElementById("recommendations").innerHTML = 
                "<p>Ошибка: ${error.message}</p>";
        });
}

function saveChatHistory() {
    localStorage.setItem("chatMessages", JSON.stringify(chatMessages));
}

function loadChatHistory() {
    let chatBox = document.getElementById("chat");
    chatMessages.forEach(msg => addMessageToChat(msg, false));
}

// Добавляем стили для индикатора загрузки
document.head.insertAdjacentHTML('beforeend', `
    <style>
    .loading {
        margin: 10px 0;
        padding: 10px;
        background-color: #f5f5f5;
        border-left: 3px solid #3498db;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    </style>
    `);

document.addEventListener("DOMContentLoaded", function () {
    const historyPanel = document.getElementById("historyPanel");
    const historyBtn = document.getElementById("historyBtn");

    // Восстанавливаем состояние панели из localStorage
    if (localStorage.getItem("historyPanelVisible") === "true") {
        historyPanel.style.display = "block";
        loadProjectHistory();
    }

    historyBtn.addEventListener("click", function (event) {
        event.preventDefault();
        if (historyPanel.style.display === "none" || historyPanel.style.display === "") {
            loadProjectHistory();
            historyPanel.style.display = "block";
            localStorage.setItem("historyPanelVisible", "true"); // Сохраняем состояние
        } else {
            historyPanel.style.display = "none";
            localStorage.setItem("historyPanelVisible", "false"); // Скрываем состояние
        }
    });
});


function loadProjectHistory() {
    fetch(`http://127.0.0.1:8000/projects/${userId}`)
        .then(response => response.json())
        .then(data => {
            let projectsList = document.getElementById("projectsList");
            projectsList.innerHTML = ""; // Очищаем список перед загрузкой
            disableVotedButtons();
            
            if (data.length === 0) {
                projectsList.innerHTML = "<p>У вас пока нет сохраненных рекомендаций.</p>";
                return;
            }

            data.forEach(project => {
                let projectDiv = document.createElement("div");
                projectDiv.className = "project-item";
                projectDiv.setAttribute("data-technology-id", project.technology_id);  // Сохраняем technology_id
                projectDiv.innerHTML = `
                    <h3>${project.project_name}</h3>
                    <p><strong>Тип:</strong> ${project.type}, <strong>Платформа:</strong> ${project.platform || "Не указано"}</p>
                    <p><strong>Бюджет:</strong> ${project.budget}, <strong>Опыт:</strong> ${project.experience_level}</p>
                    <p><strong>Рекомендация:</strong> ${project.recommendation_text || "Нет рекомендации"}</p>
                    <button class="delete-btn" data-project-id="${project.id}" style="margin-top: 10px;">Удалить</button>
                    <div class="feedback-buttons" style="margin-top: 10px;">
                        <button class="like-btn" style="font-size: 14px; padding: 6px 10px;" onclick="sendFeedback(1, '${project.technology_id}', this, event)">👍</button>
                        <button class="dislike-btn" style="font-size: 14px; padding: 6px 10px;" onclick="sendFeedback(-1, '${project.technology_id}', this, event)">👎</button>
                    </div>
                `;
                projectsList.appendChild(projectDiv);
            });

            // Добавляем обработчики удаления после загрузки
            document.querySelectorAll(".delete-btn").forEach(button => {
                button.addEventListener("click", function(event) {
                    event.preventDefault();
                    let projectId = this.getAttribute("data-project-id");
                    console.log("Удаляем проект с ID:", projectId);
                    deleteProject(projectId);
                });
            });            
        })
        .catch(error => {
            console.error("Ошибка при загрузке истории проектов:", error);
            document.getElementById("projectsList").innerHTML = "<p>Ошибка загрузки данных.</p>";
        });
}

function sendFeedback(rating, technologyId, button) {
    const userId = localStorage.getItem("userId");
    if (!userId) {
        console.error("Ошибка: userId не найден!");
        return;
    }

    // Проверяем, голосовал ли пользователь ранее
    const feedbackKey = `feedback_${userId}_${technologyId}`;
    if (localStorage.getItem(feedbackKey)) {
        console.warn("Пользователь уже голосовал за эту рекомендацию.");
        return;
    }

    fetch("http://127.0.0.1:8000/feedback/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            user_id: userId,
            technology_id: technologyId,
            rating: rating
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Фидбэк отправлен:", data.message);

        // Сохраняем в localStorage, что пользователь голосовал
        localStorage.setItem(feedbackKey, rating);

        // Отключаем кнопки
        disableFeedbackButtons(button);
    })
    .catch(error => {
        console.error("Ошибка отправки фидбэка:", error);
    });
}


function disableVotedButtons() {
    const userId = localStorage.getItem("userId");
    if (!userId) return;

    document.querySelectorAll(".project-item").forEach(item => {
        const technologyId = item.getAttribute("data-technology-id");  // Получаем technology_id
        const feedbackKey = `feedback_${userId}_${technologyId}`;  // Используем technology_id
        const storedVote = localStorage.getItem(feedbackKey);

        if (storedVote) {
            const likeBtn = item.querySelector(".like-btn");
            const dislikeBtn = item.querySelector(".dislike-btn");
            likeBtn.disabled = true;
            dislikeBtn.disabled = true;
            likeBtn.style.opacity = "0.5";
            dislikeBtn.style.opacity = "0.5";

            const feedbackMessage = document.createElement("p");
            feedbackMessage.textContent = "Спасибо за ваш отзыв!";
            feedbackMessage.style.color = "gray";
            feedbackMessage.style.marginTop = "5px";
            item.appendChild(feedbackMessage);
        }
    });
}   

function disableFeedbackButtons(button) {
    const buttonsContainer = button.parentElement;
    const likeBtn = buttonsContainer.querySelector(".like-btn");
    const dislikeBtn = buttonsContainer.querySelector(".dislike-btn");
    
    likeBtn.disabled = true;
    dislikeBtn.disabled = true;
    
    likeBtn.style.opacity = "0.5";
    dislikeBtn.style.opacity = "0.5";
    
    const feedbackMessage = document.createElement("p");
    feedbackMessage.textContent = "Спасибо за ваш отзыв!";
    feedbackMessage.style.color = "gray";
    feedbackMessage.style.marginTop = "5px";
    buttonsContainer.appendChild(feedbackMessage);
}


function loadLatestRecommendation() {
    fetch(`http://127.0.0.1:8000/projects/${userId}`)
        .then(response => response.json())
        .then(data => {
            let recommendationsDiv = document.getElementById("recommendations");
            recommendationsDiv.innerHTML = ""; // Очищаем перед добавлением новой рекомендации
            
            if (data.length === 0) {
                recommendationsDiv.innerHTML = "<p>Здесь появятся рекомендации после анализа.</p>";
                return;
            }
            
            let latestProject = data[data.length - 1]; // Берем последний добавленный проект
            let rec = document.createElement("div");
            rec.className = "recommendation-item";
            rec.setAttribute("data-technology-id", latestProject.technology_id);  // Используем technology_id
            rec.innerHTML = `
                <h3>${latestProject.project_name}</h3>
                <p><strong>Рекомендация:</strong> ${latestProject.recommendation_text}</p>
                <p><strong>Тип:</strong> ${latestProject.type}</p>
                <p><strong>Платформа:</strong> ${latestProject.platform}</p>
                <p><strong>Бюджет:</strong> ${latestProject.budget}</p>
                <p><strong>Опыт:</strong> ${latestProject.experience_level}</p>
                <div class="feedback-buttons" style="margin-top: 10px;">
                    <button class="like-btn" style="font-size: 14px; padding: 6px 10px;" onclick="sendFeedback(1, '${latestProject.technology_id}', this)">👍</button>
                    <button class="dislike-btn" style="font-size: 14px; padding: 6px 10px;" onclick="sendFeedback(-1, '${latestProject.technology_id}', this)">👎</button>
                </div>
            `;
            recommendationsDiv.appendChild(rec);
        })
        .catch(error => {
            console.error("Ошибка при получении последней рекомендации:", error);
            document.getElementById("recommendations").innerHTML = "<p>Ошибка загрузки данных.</p>";
        });
}
loadLatestRecommendation();

function deleteProject(projectId) {
    if (!projectId || projectId === "undefined") { 
        console.error("Ошибка: id не определен!", projectId);
        return;
    }

    console.log("Отправляем DELETE-запрос для id:", projectId);

    fetch(`http://127.0.0.1:8000/projects/${projectId}`, { method: "DELETE" })
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка при удалении');
            }
            return response.json();
        })
        .then(data => {
            console.log("Проект удален:", data.message);

            // Удаляем проект из DOM без перезагрузки страницы
            let projectElement = document.querySelector(`.delete-btn[data-project-id="${projectId}"]`).closest('.project-item');
            if (projectElement) {
                projectElement.remove(); // Удаляем HTML-элемент
                console.log("Удален элемент из DOM:", projectId);
            } else {
                console.error("Не найден элемент для удаления с ID:", projectId);
            }
            
            // Проверяем, остались ли еще проекты
            let remainingProjects = document.querySelectorAll('.project-item');
            if (remainingProjects.length === 0) {
                document.getElementById("projectsList").innerHTML = "<p>У вас пока нет сохраненных рекомендаций.</p>";
            }
        })
        .catch(error => {
            console.error("Ошибка при удалении проекта:", error);
        });
}

// Сохранение в PDF
function addPDFListener() {
    const saveBtn = document.getElementById("savePdfBtn");
    if (saveBtn) {
        saveBtn.addEventListener("click", () => {
            // Проверяем, есть ли рекомендации в блоке чата или в контейнере рекомендаций
            const recommendationsDiv = document.getElementById("recommendations");
            const hasRecommendationsInDiv = recommendationsDiv && 
                recommendationsDiv.querySelector(".recommendation-item");
                
            // Проверяем, есть ли слово "рекомендация" в сообщениях чата
            const chatMessages = document.querySelectorAll(".chat-message.system p");
            let hasRecommendationsInChat = false;
            
            for (let message of chatMessages) {
                if (message.textContent.includes("Вот рекомендация:")) {
                    hasRecommendationsInChat = true;
                    break;
                }
            }
            
            // Если нет рекомендаций ни в контейнере, ни в чате, показываем сообщение
            if (!hasRecommendationsInDiv && !hasRecommendationsInChat) {
                addMessageToChat({
                    type: "system",
                    text: "Невозможно сохранить PDF. Сначала получите рекомендации, завершив опрос."
                }, true);
                saveChatHistory();
                return;
            }
            
            // Если рекомендации есть только в чате, но нет в контейнере,
            // пробуем обновить рекомендации перед созданием PDF
            if (!hasRecommendationsInDiv && hasRecommendationsInChat) {
                // Показываем сообщение о получении данных
                addMessageToChat({
                    type: "system",
                    text: "Загружаем рекомендации для PDF..."
                }, true);
                saveChatHistory();
                
                // Запускаем получение рекомендаций
                try {
                    getRecommendation();
                } catch (e) {
                    console.error("Ошибка при обновлении рекомендаций:", e);
                }
            }
            
            // Получаем userId из sessionStorage
            const userId = localStorage.getItem("userId");
            
            // Показываем индикатор загрузки
            document.getElementById("recommendations").innerHTML += "<p class='loading'>Создание PDF...</p>";
            
            // Делаем запрос к backend для получения PDF
            fetch(`http://127.0.0.1:8000/export_pdf/${userId}`, {
                method: "GET",
                responseType: "blob" // Важно для получения бинарных данных
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка при генерации PDF');
                }
                return response.blob();
            })
            .then(blob => {
                // Создаем URL для скачивания
                const url = window.URL.createObjectURL(blob);
                
                // Создаем временную ссылку для скачивания
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'recommendations.pdf';
                
                // Добавляем ссылку в DOM и запускаем скачивание
                document.body.appendChild(a);
                a.click();
                
                // Удаляем ссылку и освобождаем URL
                window.URL.revokeObjectURL(url);
                a.remove();
                
                // Удаляем индикатор загрузки
                const loadingElement = document.querySelector('.loading');
                if (loadingElement) {
                    loadingElement.remove();
                }
                
                // Показываем сообщение об успехе
                addMessageToChat({
                    type: "system",
                    text: "PDF успешно сохранен!"
                }, true);
                saveChatHistory();
            })
            .catch(error => {
                console.error("Ошибка при скачивании PDF:", error);
                
                // Удаляем индикатор загрузки
                const loadingElement = document.querySelector('.loading');
                if (loadingElement) {
                    loadingElement.remove();
                }
                
                // Показываем сообщение об ошибке
                addMessageToChat({
                    type: "system",
                    text: "Произошла ошибка при сохранении PDF. Пожалуйста, попробуйте позже."
                }, true);
                saveChatHistory();
            });
        });
    } else {
        console.error("Кнопка 'Сохранить в PDF' не найдена");
    }
}

// Слушатели событий
document.getElementById("sendBtn").addEventListener("click", function(event) {
    event.preventDefault();
    sendMessage();
});

document.getElementById("userInput").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

document.querySelectorAll('.sidebar a').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();
        console.log('Клик по ссылке предотвращен');
    });
});

document.getElementById("restartFormBtn").addEventListener("click", function () {
    localStorage.removeItem('chatMessages');
    fetch(`http://127.0.0.1:8000/restart_session/${userId}`, { method: "POST" })
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка при перезапуске сессии');
            }
            return response.json();
        })
        .then(data => {
            console.log("Форма перезапущена:", data.message);

            // Очищаем чат
            chatMessages = []; // Очищаем массив сообщений
            document.getElementById("chat").innerHTML = ""; // Удаляем все сообщения

            document.getElementById("chat").innerHTML = `<div class="chat-message system">
                <p>Привет! Давайте начнем. Какой тип приложения вы разрабатываете?</p>
                <small style="color: gray;">Например: нативное, кроссплатформенное или гибридное</small>
            </div>`;
        })
        .catch(error => {
            console.error("Ошибка при перезапуске формы:", error);
            addMessageToChat({
                type: "system",
                text: "Ошибка при перезапуске формы. Пожалуйста, попробуйте еще раз."
            }, true);
        });
});

// Функция для показа модального окна
function showEmailModal() {
    const modal = document.getElementById("emailModal");
    modal.classList.add("active");
    document.getElementById("emailInput").focus();
}

// Функция для скрытия модального окна
function hideEmailModal() {
    const modal = document.getElementById("emailModal");
    modal.classList.remove("active");
}

// Функция для отправки email
function sendEmail() {
    const emailInput = document.getElementById("emailInput");
    const userEmail = emailInput.value.trim();
    
    // Валидация email
    if (!userEmail || !userEmail.includes("@") || !userEmail.includes(".")) {
        alert("Пожалуйста, введите корректный email адрес");
        return;
    }
    
    hideEmailModal();
    
    addMessageToChat({
        type: "system",
        text: "Отправляем рекомендации на вашу почту..."
    }, true);

    // Добавляем таймаут для запроса (например, 10 секунд)
    const timeout = 10000; // 10 секунд
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
            reject(new Error("Превышено время ожидания отправки письма"));
        }, timeout);
    });

    Promise.race([
        fetch("http://127.0.0.1:8000/send_email/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                user_email: userEmail, 
                user_id: userId  
            })
        }),
        timeoutPromise
    ])
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success || data.message === "Рекомендации отправлены на email!") {
            addMessageToChat({
                type: "system",
                text: `Рекомендации успешно отправлены на вашу почту: ${userEmail}`
            }, true);
        } else {
            throw new Error(data.message || "Не удалось отправить письмо");
        }
        emailInput.value = "";
    })
    .catch(error => {
        console.error("Ошибка отправки email:", error);
        addMessageToChat({
            type: "system",
            text: `Произошла ошибка при отправке рекомендаций на почту: ${error.message}. Пожалуйста, попробуйте позже или проверьте подключение к интернету.`
        }, true);
    });
}

// Инициализация обработчиков событий для модального окна
function initEmailModal() {
    document.getElementById("sendEmailBtn").addEventListener("click", showEmailModal);
    document.getElementById("cancelEmailBtn").addEventListener("click", hideEmailModal);
    document.getElementById("submitEmailBtn").addEventListener("click", sendEmail);
    
    // Закрытие по клику вне окна
    document.getElementById("emailModal").addEventListener("click", function(e) {
        if (e.target === this) {
            hideEmailModal();
        }
    });
    
    // Закрытие по Esc
    document.addEventListener("keydown", function(e) {
        const modal = document.getElementById("emailModal");
        if (e.key === "Escape" && modal.classList.contains("active")) {
            hideEmailModal();
        }
    });
}

// Загрузка после DOM
window.addEventListener("DOMContentLoaded", () => {
    addPDFListener();
    loadChatHistory();
    initEmailModal();
    // loadRecommendations();
});


