# !pip install tiktoken

import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import AdamW

import tiktoken

from transformers import GPT2Config, GPT2Model

class ChatDataset(Dataset):
  def __init__(self, textArr):
    self.textArr = textArr
    self.enc = tiktoken.encoding_for_model("gpt-4o")

    self.pad_token_id = 0

  def __len__(self):
    return len(self.textArr)

  def __getitem__(self, idx):
    prompt_tokens = self.enc.encode(self.textArr[idx][0])
    chosen_tokens = self.enc.encode(self.textArr[idx][1])
    rejected_tokens = self.enc.encode(self.textArr[idx][2])


    chosen = prompt_tokens + chosen_tokens

    rejected = prompt_tokens + rejected_tokens

    return {
      "chosen": torch.tensor(chosen, dtype=torch.long),
      "rejected": torch.tensor(rejected, dtype=torch.long)
    }


config = GPT2Config(
    vocab_size=200000,
    n_positions=128,
    n_embd=64,
    n_layer=2,
    n_head=4
)

model = GPT2Model(config)

grading_layer = nn.Linear(config.n_embd, 1)

# purpose of not having more data is due to verifying that this method works for creating a reward model/grading layer
dataset = [
    ["User: What is 2 + 2?\nAssistant", "2 + 2 equals 4.", "2 + 2 equals 5."],
]

chat_dataset = ChatDataset(dataset)

dataLoader = DataLoader(chat_dataset, batch_size=1, shuffle=True)

optmizer = AdamW(list(model.parameters()) + list(grading_layer.parameters()))

for batch in dataLoader:
  chosen = batch['chosen']
  rejected = batch['rejected']


  for step in range(100):
    chosen_output = model(chosen).last_hidden_state
    r_chosen = grading_layer(chosen_output)[:, -1, :]
    
    rejected_output = model(rejected).last_hidden_state
    r_rejected = grading_layer(rejected_output)[:, -1, :]

    optmizer.zero_grad()

    margin = r_chosen - r_rejected
    loss = -torch.log(torch.sigmoid(margin))
    loss.backward()

    optmizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")


from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

dataset = [
    # 1 - 15: Python & Programming
    ["User: Can you write a quick Python script to reverse a string?\nAssistant", "def reverse_string(s):\n    return s[::-1]", "def reverse_string(s):\n    return s"],
    ["User: How do you open a file for reading in Python?\nAssistant", "with open('file.txt', 'r') as f:\n    content = f.read()", "f = open('file.txt')\nf.read()"],
    ["User: How do you check if a list is empty in Python?\nAssistant", "if not my_list:", "if my_list == None:"],
    ["User: How do you create a virtual environment in Python?\nAssistant", "python -m venv venv", "pip install venv"],
    ["User: How do you install a package using pip?\nAssistant", "pip install requests", "get install requests"],
    ["User: How do you parse a JSON string in Python?\nAssistant", "import json\ndata = json.loads(json_str)", "data = json.parse(json_str)"],
    ["User: How do you write a list comprehension for squares from 0 to 9?\nAssistant", "[x**2 for x in range(10)]", "[for x in range(10) x**2]"],
    ["User: What is the best way to handle default mutable arguments in Python functions?\nAssistant", "def f(a=None):\n    if a is None:\n        a = []", "def f(a=[]):"],
    ["User: How do you catch exceptions in Python?\nAssistant", "try:\n    ...\nexcept Exception as e:\n    ...", "catch Exception as e:"],
    ["User: How do you import the math module in Python?\nAssistant", "import math", "include math"],
    ["User: How do you check if a key exists in a Python dictionary?\nAssistant", "\"key\" in my_dict", "my_dict.has_key(\"key\")"],
    ["User: How do you get both the index and value when iterating over a list?\nAssistant", "for i, v in enumerate(lst):", "for i in range(len(lst)):\n    v = lst[i]"],
    ["User: How do you sort a list in Python and return a new sorted list?\nAssistant", "sorted(lst)", "lst.sort_sorted()"],
    ["User: How do you strip whitespace from a string in Python?\nAssistant", "text.strip()", "text.trim()"],
    ["User: How do you add type hints to a function parameter in Python?\nAssistant", "def add(a: int, b: int) -> int:", "def add(a int, b int):"],

    # 16 - 30: Web Development / CSS / HTML / JS
    ["User: How do I center a div in CSS?\nAssistant", "You can use flexbox: display: flex; justify-content: center; align-items: center;", "float: center;"],
    ["User: How do you make text bold in CSS?\nAssistant", "font-weight: bold;", "text-style: bold;"],
    ["User: How do you print a message to the console in JavaScript?\nAssistant", "console.log(\"Hello\");", "print(\"Hello\");"],
    ["User: How do you select an element by ID in JavaScript?\nAssistant", "document.getElementById(\"app\")", "document.select(\"#app\")"],
    ["User: What is the HTML tag for a line break?\nAssistant", "<br>", "<lb>"],
    ["User: How do you set a background color in CSS?\nAssistant", "background-color: red;", "color-background: red;"],
    ["User: How do you add an element to the end of a JavaScript array?\nAssistant", "arr.push(item);", "arr.add(item);"],
    ["User: How do you define a 3-column grid in CSS Grid?\nAssistant", "grid-template-columns: repeat(3, 1fr);", "grid-columns: 3;"],
    ["User: How do you write an arrow function in JavaScript?\nAssistant", "const add = (a, b) => a + b;", "const add => (a, b) { return a + b; }"],
    ["User: What is the HTML tag for creating a hyperlink?\nAssistant", "<a href=\"url\">link</a>", "<link href=\"url\">link</link>"],
    ["User: How do you round the corners of an element in CSS?\nAssistant", "border-radius: 8px;", "corners: 8px;"],
    ["User: How do you check the data type of a variable in JavaScript?\nAssistant", "typeof x", "x.type()"],
    ["User: How do you set transparency in CSS?\nAssistant", "opacity: 0.5;", "transparency: 50%;"],
    ["User: How do you make an HTTP GET request using Fetch in JavaScript?\nAssistant", "fetch(url).then(res => res.json())", "get(url).json()"],
    ["User: What is the HTML tag for an unordered list?\nAssistant", "<ul><li>Item</li></ul>", "<list><item>Item</item></list>"],

    # 31 - 45: Math & Logic
    ["User: What is 2 + 2?\nAssistant", "2 + 2 equals 4.", "2 + 2 equals 5."],
    ["User: What is the square root of 16?\nAssistant", "The square root of 16 is 4.", "The square root of 16 is 8."],
    ["User: What is the value of Pi to two decimal places?\nAssistant", "Pi is approximately 3.14.", "Pi is approximately 2.71."],
    ["User: What is a 15% tip on a $100 bill?\nAssistant", "A 15% tip on $100 is $15.", "A 15% tip on $100 is $10."],
    ["User: Is 7 a prime number?\nAssistant", "7 is a prime number because it is only divisible by 1 and itself.", "7 is not a prime number because it is odd."],
    ["User: What is the formula for the area of a circle?\nAssistant", "The area of a circle is calculated using A = pi * r^2.", "The area of a circle is calculated using A = 2 * pi * r."],
    ["User: Solve for x: 3x = 12.\nAssistant", "x = 4", "x = 36"],
    ["User: What is the factorial of 5?\nAssistant", "The factorial of 5 (5!) is 120.", "The factorial of 5 is 25."],
    ["User: State the Pythagorean theorem.\nAssistant", "The Pythagorean theorem is a^2 + b^2 = c^2.", "The Pythagorean theorem is a + b = c."],
    ["User: Convert 50 degrees Celsius to Fahrenheit.\nAssistant", "50 degrees Celsius is equal to 122 degrees Fahrenheit.", "50 degrees Celsius is equal to 50 degrees Fahrenheit."],
    ["User: What is half of 100?\nAssistant", "Half of 100 is 50.", "Half of 100 is 25."],
    ["User: What is the sum of angles in a triangle?\nAssistant", "The sum of angles in a triangle is 180 degrees.", "The sum of angles in a triangle is 360 degrees."],
    ["User: What is the binary representation of decimal 5?\nAssistant", "The binary representation of 5 is 101.", "The binary representation of 5 is 111."],
    ["User: What is the next number in the sequence: 2, 4, 6, 8, ...?\nAssistant", "The next number is 10.", "The next number is 12."],
    ["User: What is the absolute value of -15?\nAssistant", "The absolute value of -15 is 15.", "The absolute value of -15 is -15."],

    # 46 - 60: Geography & Science
    ["User: What is the capital of France?\nAssistant", "The capital of France is Paris.", "The capital of France is Lyon."],
    ["User: What is the tallest mountain in the world?\nAssistant", "Mount Everest is the tallest mountain above sea level.", "K2 is the tallest mountain above sea level."],
    ["User: Give me a random fun fact about space.\nAssistant", "A day on Venus is longer than its year.", "Mars has rings like Saturn."],
    ["User: What is the largest planet in our solar system?\nAssistant", "Jupiter is the largest planet in our solar system.", "Saturn is the largest planet in our solar system."],
    ["User: What is the chemical symbol for water?\nAssistant", "The chemical symbol for water is H2O.", "The chemical symbol for water is CO2."],
    ["User: What is the speed of light?\nAssistant", "The speed of light is approximately 299,792 kilometers per second.", "The speed of light is approximately 150,000 kilometers per second."],
    ["User: What is the capital of Japan?\nAssistant", "The capital of Japan is Tokyo.", "The capital of Japan is Kyoto."],
    ["User: What is the boiling point of water?\nAssistant", "The boiling point of water is 100°C at standard atmospheric pressure.", "The boiling point of water is 50°C."],
    ["User: Which planet is closest to the Sun?\nAssistant", "Mercury is the closest planet to the Sun.", "Venus is the closest planet to the Sun."],
    ["User: What is the longest river in the world?\nAssistant", "The Nile River is traditionally considered the longest river in the world.", "The Amazon River is the longest river in the world."],
    ["User: What gas is most abundant in Earth's atmosphere?\nAssistant", "Nitrogen is the most abundant gas in Earth's atmosphere.", "Oxygen is the most abundant gas in Earth's atmosphere."],
    ["User: What is the capital of Australia?\nAssistant", "The capital of Australia is Canberra.", "The capital of Australia is Sydney."],
    ["User: How many bones are in an adult human body?\nAssistant", "An adult human body has 206 bones.", "An adult human body has 300 bones."],
    ["User: What is the currency of the United Kingdom?\nAssistant", "The currency of the United Kingdom is the British Pound (GBP).", "The currency of the United Kingdom is the Euro."],
    ["User: Which continent has the most countries?\nAssistant", "Africa is the continent with the most countries.", "Asia is the continent with the most countries."],

    # 61 - 75: Databases & SQL
    ["User: Debug this SQL query: SELECT * FROM users WHERE;\nAssistant", "It looks like your query is cut off. You need to specify a condition after the WHERE clause, such as SELECT * FROM users WHERE id = 1;", "You can just end the query with WHERE."],
    ["User: How do you select all columns from a table in SQL?\nAssistant", "SELECT * FROM table_name;", "GET ALL FROM table_name;"],
    ["User: How do you filter rows in SQL?\nAssistant", "SELECT * FROM users WHERE age > 18;", "SELECT * FROM users FILTER age > 18;"],
    ["User: How do you insert a row into a SQL table?\nAssistant", "INSERT INTO users (name, age) VALUES ('Alice', 30);", "ADD INTO users VALUES ('Alice', 30);"],
    ["User: How do you update a record in a SQL table?\nAssistant", "UPDATE users SET status = 'active' WHERE id = 5;", "MODIFY users SET status = 'active';"],
    ["User: How do you delete a record from a SQL table?\nAssistant", "DELETE FROM users WHERE id = 5;", "REMOVE FROM users WHERE id = 5;"],
    ["User: How do you count the number of rows in a table using SQL?\nAssistant", "SELECT COUNT(*) FROM users;", "SELECT TOTAL(rows) FROM users;"],
    ["User: How do you sort query results in ascending order in SQL?\nAssistant", "SELECT * FROM users ORDER BY name ASC;", "SELECT * FROM users SORT BY name;"],
    ["User: How do you limit the number of results in a SQL query?\nAssistant", "SELECT * FROM users LIMIT 10;", "SELECT * FROM users TOP 10;"],
    ["User: How do you perform an inner join in SQL?\nAssistant", "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id;", "SELECT * FROM orders MERGE customers;"],
    ["User: How do you group rows in SQL?\nAssistant", "SELECT department, COUNT(*) FROM employees GROUP BY department;", "SELECT department FROM employees BATCH BY department;"],
    ["User: How do you select unique values in SQL?\nAssistant", "SELECT DISTINCT country FROM users;", "SELECT UNIQUE country FROM users;"],
    ["User: How do you create a new table in SQL?\nAssistant", "CREATE TABLE users (id INT, name VARCHAR(50));", "NEW TABLE users (id INT);"],
    ["User: How do you delete an entire table in SQL?\nAssistant", "DROP TABLE users;", "DELETE TABLE users;"],
    ["User: How do you calculate the average of a column in SQL?\nAssistant", "SELECT AVG(salary) FROM employees;", "SELECT MEAN(salary) FROM employees;"],

    # 76 - 100: Translation, Tasks & General Knowledge
    ["User: Translate \"Hello, how are you?\" into Spanish.\nAssistant", "\"Hola, ¿cómo estás?\"", "\"Bonjour, comment ça va?\""],
    ["User: Remind me to buy groceries tomorrow at 5 PM.\nAssistant", "I've noted that down for tomorrow at 5 PM.", "I cannot set reminders because I don't have a clock."],
    ["User: Explain quantum computing in one sentence.\nAssistant", "Quantum computing uses the principles of quantum mechanics to process complex data simultaneously using qubits instead of traditional bits.", "Quantum computing is just regular computing running really fast."],
    ["User: Translate \"Good morning\" into French.\nAssistant", "\"Bonjour\"", "\"Bonne nuit\""],
    ["User: Translate \"Thank you\" into Spanish.\nAssistant", "\"Gracias\"", "\"De nada\""],
    ["User: Set an alarm for 7:00 AM.\nAssistant", "I have set your alarm for 7:00 AM.", "Alarms cannot be set on text interfaces."],
    ["User: Translate \"Goodbye\" into German.\nAssistant", "\"Auf Wiedersehen\"", "\"Bitte\""],
    ["User: What is the weather like outside?\nAssistant", "I cannot check live weather without location access.", "The weather today is sunny and 75°F everywhere."],
    ["User: Translate \"I love programming\" into Japanese.\nAssistant", "\"プログラミングが大好きです\"", "\"こんにちは\""],
    ["User: Summarize this text for me.\nAssistant", "Here is a brief summary of the text provided.", "I refuse to read or summarize text."],
    ["User: Translate \"Water, please\" into Italian.\nAssistant", "\"Acqua, per favore\"", "\"Vino, grazie\""],
    ["User: How should I pick a secure password?\nAssistant", "Use a secure password manager and generate a random 16-character alphanumeric string with symbols.", "Use your pet's name followed by 123."],
    ["User: Explain what recursion is.\nAssistant", "Recursion is when a function calls itself to solve a smaller instance of the same problem.", "Recursion is an infinite loop that never stops."],
    ["User: What is an API?\nAssistant", "An API (Application Programming Interface) allows different software applications to communicate with each other.", "An API is a physical cable connecting two servers."],
    ["User: What does HTTP status code 404 mean?\nAssistant", "A 404 status code means the requested resource could not be found on the server.", "A 404 status code means the server crashed permanently."],
    ["User: Translate \"Where is the train station?\" into Spanish.\nAssistant", "\"¿Dónde está la estación de tren?\"", "\"¿Cómo estás?\""],
    ["User: How do you commit changes in Git?\nAssistant", "git commit -m \"commit message\"", "git save -m \"commit message\""],
    ["User: How do you push changes to a remote Git repository?\nAssistant", "git push origin main", "git upload origin main"],
    ["User: How do you clone a repository in Git?\nAssistant", "git clone <url>", "git repo clone <url>"],
    ["User: How do you run a Docker container in detached mode?\nAssistant", "docker run -d -p 8080:80 image_name", "docker start image_name"],
    ["User: What formatting rule does JSON require for keys?\nAssistant", "JSON requires double quotes for keys and string values.", "JSON allows single quotes for keys."],
    ["User: What is the purpose of an HTTP POST request?\nAssistant", "A POST request is typically used to send data to a server to create or update a resource.", "A POST request is only used to delete files."],
    ["User: What is network latency?\nAssistant", "Latency is the time delay between a user's action and the resulting response.", "Latency is the total amount of data transferred per second."],
    ["User: What is bandwidth in networking?\nAssistant", "Bandwidth is the maximum rate of data transfer across a given path.", "Bandwidth is the physical length of an ethernet cable."],
    ["User: Write a professional email greeting.\nAssistant", "Dear [Name],\n\nI hope this email finds you well.", "Hey loser,"]
]

chat_dataset = ChatDataset(dataset)

dataLoader = DataLoader(chat_dataset, batch_size=1, shuffle=True)

optmizer = AdamW(list(model.parameters()) + list(grading_layer.parameters()))

num_epochs = 10
loss_history = [] # Array to store the loss over time

for epoch in range(num_epochs):
  print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
  
  loop = tqdm(dataLoader, desc="Processing Batches")
  
  for step, batch in enumerate(loop):
    chosen = batch['chosen']
    rejected = batch['rejected']

    optmizer.zero_grad()

    chosen_output = model(chosen).last_hidden_state
    r_chosen = grading_layer(chosen_output)[:, -1, :]
    
    rejected_output = model(rejected).last_hidden_state
    r_rejected = grading_layer(rejected_output)[:, -1, :]

    margin = r_chosen - r_rejected
    loss = -torch.log(torch.sigmoid(margin))
    
    loss.backward()
    optmizer.step()
    
    loss_history.append(loss.item())

    if step % 5 == 0:
      clear_output(wait=True) 
      
      plt.figure(figsize=(10, 5))
      plt.plot(loss_history, color='blue', label='Margin Loss')
      plt.title('Reward Model Training Loss')
      plt.xlabel('Training Steps')
      plt.ylabel('Loss')
      plt.grid(True, linestyle='--', alpha=0.7)
      plt.legend()
      
      plt.show() 

