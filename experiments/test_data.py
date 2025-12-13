"""
Synthetic Test Dataset for Smart Router Experiments

This module provides test queries categorized by type with expected behaviors
and ground truth answers for evaluation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TestQuery:
    """A test query with metadata for evaluation"""
    query: str
    query_type: str
    expected_route: str  # "local" or "remote"
    ground_truth: str  # Expected answer for accuracy evaluation
    context: List[str]  # Optional context documents
    difficulty: str  # "easy", "medium", "hard"


# Lookup queries - should route to local
LOOKUP_QUERIES = [
    TestQuery(
        query="What is a cartridge",
        query_type="lookup",
        expected_route="local",
        ground_truth="A cartridge is a container or case that holds a substance, device, or material. Common types include ink cartridges for printers, game cartridges for video game consoles, and ammunition cartridges for firearms.",
        context=[
            "A cartridge is a container that holds ink, toner, or other materials for printers and copiers.",
            "Game cartridges are physical media containing video game data, used in older gaming consoles.",
            "Ammunition cartridges consist of a casing, primer, propellant, and projectile.",
            "Printer cartridges can be inkjet or laser toner types, with varying capacities and costs."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define machine learning",
        query_type="lookup",
        expected_route="local",
        ground_truth="Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions or predictions.",
        context=[
            "Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
            "ML algorithms can be supervised (labeled data), unsupervised (unlabeled data), or reinforcement learning.",
            "Common ML applications include image recognition, natural language processing, and recommendation systems.",
            "Deep learning is a subset of ML using neural networks with multiple layers."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the meaning of blockchain",
        query_type="lookup",
        expected_route="local",
        ground_truth="Blockchain is a distributed digital ledger technology that records transactions across multiple computers in a way that makes the records difficult to alter retroactively. Each block contains transaction data and is cryptographically linked to the previous block.",
        context=[
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks.",
            "Each block contains a cryptographic hash of the previous block, timestamp, and transaction data.",
            "Blockchain is decentralized and managed by a peer-to-peer network, making it resistant to modification.",
            "Bitcoin and Ethereum are popular blockchain-based cryptocurrencies."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Explain photosynthesis",
        query_type="lookup",
        expected_route="local",
        ground_truth="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
        context=[
            "Photosynthesis is the process plants use to convert light energy into chemical energy stored in glucose.",
            "The process occurs in chloroplasts using the pigment chlorophyll to capture light energy.",
            "The light-dependent reactions occur in thylakoid membranes, producing ATP and NADPH.",
            "The Calvin cycle uses ATP and NADPH to convert CO2 into glucose in the stroma."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is GDP",
        query_type="lookup",
        expected_route="local",
        ground_truth="GDP (Gross Domestic Product) is the total monetary value of all finished goods and services produced within a country's borders in a specific time period. It is a key indicator of a country's economic health.",
        context=[
            "GDP measures the size and health of an economy.",
            "It includes consumption, investment, government spending, and net exports.",
            "GDP can be measured in nominal terms or adjusted for inflation (real GDP)."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define neural network",
        query_type="lookup",
        expected_route="local",
        ground_truth="A neural network is a computing system inspired by biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that process and transmit information to learn patterns from data.",
        context=[
            "Neural networks consist of input, hidden, and output layers.",
            "Each connection has a weight that is adjusted during training.",
            "They are used in deep learning for tasks like image recognition and natural language processing."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is DNA",
        query_type="lookup",
        expected_route="local",
        ground_truth="DNA (Deoxyribonucleic Acid) is a molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms. It consists of two strands forming a double helix structure.",
        context=[
            "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
            "The sequence of these bases determines genetic information.",
            "DNA is found in the nucleus of cells and can replicate itself."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Explain what HTTP stands for",
        query_type="lookup",
        expected_route="local",
        ground_truth="HTTP stands for HyperText Transfer Protocol. It is the foundation of data communication on the World Wide Web, defining how messages are formatted and transmitted between web browsers and servers.",
        context=[
            "HTTP is an application-layer protocol used for transmitting hypermedia documents.",
            "HTTPS is the secure version using encryption.",
            "Common HTTP methods include GET, POST, PUT, and DELETE."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the capital of France",
        query_type="lookup",
        expected_route="local",
        ground_truth="Paris",
        context=[
            "Paris is the capital and largest city of France.",
            "It is located in north-central France on the Seine River."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Who wrote 'Romeo and Juliet'",
        query_type="lookup",
        expected_route="local",
        ground_truth="William Shakespeare",
        context=[
            "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            "It is among Shakespeare's most popular plays during his lifetime."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the chemical symbol for Gold",
        query_type="lookup",
        expected_route="local",
        ground_truth="Au",
        context=[
            "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79.",
            "It is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define 'metaphor'",
        query_type="lookup",
        expected_route="local",
        ground_truth="A figure of speech in which a word or phrase is applied to an object or action to which it is not literally applicable.",
        context=[
            "A metaphor is a figure of speech that describes an object or action in a way that isn't literally true.",
            "It helps explain an idea or make a comparison."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the largest planet in our solar system",
        query_type="lookup",
        expected_route="local",
        ground_truth="Jupiter",
        context=[
            "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
            "It is a gas giant with a mass one-thousandth that of the Sun."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What does 'CPU' stand for",
        query_type="lookup",
        expected_route="local",
        ground_truth="Central Processing Unit",
        context=[
            "A central processing unit (CPU), also called a central processor, main processor or just processor, is the electronic circuitry that executes instructions comprising a computer program.",
            "The CPU performs basic arithmetic, logic, controlling, and input/output (I/O) operations."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Who painted the Mona Lisa",
        query_type="lookup",
        expected_route="local",
        ground_truth="Leonardo da Vinci",
        context=[
            "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
            "It is considered an archetypal masterpiece of the Italian Renaissance."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the boiling point of water",
        query_type="lookup",
        expected_route="local",
        ground_truth="100 degrees Celsius or 212 degrees Fahrenheit at sea level",
        context=[
            "The boiling point of water is 100 °C (212 °F) at standard pressure (1 atmosphere).",
            "Boiling point decreases as altitude increases."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define 'osmosis'",
        query_type="lookup",
        expected_route="local",
        ground_truth="The movement of water molecules through a semipermeable membrane from a region of lower solute concentration to a region of higher solute concentration.",
        context=[
            "Osmosis is the spontaneous net movement or diffusion of solvent molecules through a selectively permeable membrane.",
            "It occurs from a region of high water potential to a region of low water potential."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the speed of light",
        query_type="lookup",
        expected_route="local",
        ground_truth="Approximately 299,792,458 meters per second",
        context=[
            "The speed of light in vacuum, commonly denoted c, is a universal physical constant.",
            "Its exact value is defined as 299,792,458 metres per second."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Who discovered penicillin",
        query_type="lookup",
        expected_route="local",
        ground_truth="Alexander Fleming",
        context=[
            "Alexander Fleming was a Scottish physician and microbiologist.",
            "His best-known discoveries are the enzyme lysozyme in 1923 and the world's first broadly effective antibiotic substance benzylpenicillin (Penicillin G) from the mould Penicillium rubens in 1928."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is a 'byte'",
        query_type="lookup",
        expected_route="local",
        ground_truth="A unit of digital information that typically consists of eight bits.",
        context=[
            "The byte is a unit of digital information that most commonly consists of eight bits.",
            "Historically, the byte was the number of bits used to encode a single character of text in a computer."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the currency of Japan",
        query_type="lookup",
        expected_route="local",
        ground_truth="Japanese Yen",
        context=[
            "The yen is the official currency of Japan.",
            "It is the third most traded currency in the foreign exchange market after the United States dollar and the Euro."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define 'algorithm'",
        query_type="lookup",
        expected_route="local",
        ground_truth="A process or set of rules to be followed in calculations or other problem-solving operations, especially by a computer.",
        context=[
            "In mathematics and computer science, an algorithm is a finite sequence of well-defined instructions.",
            "Algorithms are used for solving a class of specific problems or to perform a computation."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the main gas in Earth's atmosphere",
        query_type="lookup",
        expected_route="local",
        ground_truth="Nitrogen",
        context=[
            "The atmosphere of Earth is composed of nitrogen (about 78%), oxygen (about 21%), argon (about 0.9%), carbon dioxide (0.04%) and other gases in trace amounts.",
            "Nitrogen is the most abundant gas in the atmosphere."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Who invented the telephone",
        query_type="lookup",
        expected_route="local",
        ground_truth="Alexander Graham Bell",
        context=[
            "Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer.",
            "He is credited with patenting the first practical telephone."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the currency of the UK",
        query_type="lookup",
        expected_route="local",
        ground_truth="Pound Sterling",
        context=[
            "The pound sterling (GBP) is the official currency of the United Kingdom.",
            "It is the oldest currency in continuous use."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the capital of Australia",
        query_type="lookup",
        expected_route="local",
        ground_truth="Canberra",
        context=[
            "Canberra is the capital city of Australia.",
            "It is located at the northern end of the Australian Capital Territory."
        ],
        difficulty="easy"
    ),
    TestQuery(
        query="Define 'photosynthesis'",
        query_type="lookup",
        expected_route="local",
        ground_truth="Process by which plants use sunlight to synthesize foods",
        context=[
            "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy."
        ],
        difficulty="easy"
    ),
]

# Math queries - should route to local
MATH_QUERIES = [
    TestQuery(
        query="Calculate 15 + 27",
        query_type="math",
        expected_route="local",
        ground_truth="42",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the average of 10, 20, and 30",
        query_type="math",
        expected_route="local",
        ground_truth="20",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Compute 144 divided by 12",
        query_type="math",
        expected_route="local",
        ground_truth="12",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is 25% of 80",
        query_type="math",
        expected_route="local",
        ground_truth="20",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Calculate the square root of 169",
        query_type="math",
        expected_route="local",
        ground_truth="13",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is 7 multiplied by 8",
        query_type="math",
        expected_route="local",
        ground_truth="56",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="If a rectangle has length 12 and width 5, what is its area",
        query_type="math",
        expected_route="local",
        ground_truth="60",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Convert 98.6 degrees Fahrenheit to Celsius",
        query_type="math",
        expected_route="local",
        ground_truth="37 degrees Celsius",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Solve for x: 2x + 5 = 15",
        query_type="math",
        expected_route="local",
        ground_truth="x = 5",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is 15% of 200",
        query_type="math",
        expected_route="local",
        ground_truth="30",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Calculate the area of a circle with radius 3",
        query_type="math",
        expected_route="local",
        ground_truth="Approximately 28.27",
        context=["Area = pi * r^2", "pi is approximately 3.14159"],
        difficulty="medium"
    ),
    TestQuery(
        query="What is the prime factorization of 60",
        query_type="math",
        expected_route="local",
        ground_truth="2^2 * 3 * 5",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="If a train travels 60 miles in 1.5 hours, what is its average speed",
        query_type="math",
        expected_route="local",
        ground_truth="40 mph",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Compute 2 to the power of 8",
        query_type="math",
        expected_route="local",
        ground_truth="256",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the volume of a cube with side length 4",
        query_type="math",
        expected_route="local",
        ground_truth="64",
        context=["Volume = side^3"],
        difficulty="easy"
    ),
    TestQuery(
        query="Reduce the fraction 18/24 to lowest terms",
        query_type="math",
        expected_route="local",
        ground_truth="3/4",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the sum of angles in a triangle",
        query_type="math",
        expected_route="local",
        ground_truth="180 degrees",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Calculate the hypotenuse of a right triangle with legs 3 and 4",
        query_type="math",
        expected_route="local",
        ground_truth="5",
        context=["Pythagorean theorem: a^2 + b^2 = c^2"],
        difficulty="easy"
    ),
    TestQuery(
        query="What is 100 factorial divided by 99 factorial",
        query_type="math",
        expected_route="local",
        ground_truth="100",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="If you buy 3 items at $5.50 each, what is the total cost",
        query_type="math",
        expected_route="local",
        ground_truth="$16.50",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the binary representation of 10",
        query_type="math",
        expected_route="local",
        ground_truth="1010",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Solve for y: 3y - 9 = 0",
        query_type="math",
        expected_route="local",
        ground_truth="y = 3",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the greatest common divisor of 12 and 18",
        query_type="math",
        expected_route="local",
        ground_truth="6",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="What is the cube root of 27",
        query_type="math",
        expected_route="local",
        ground_truth="3",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Calculate 12 squared",
        query_type="math",
        expected_route="local",
        ground_truth="144",
        context=[],
        difficulty="easy"
    ),
]

# Extract queries - moderate difficulty
EXTRACT_QUERIES = [
    TestQuery(
        query="Extract all numbers from the text",
        query_type="extract",
        expected_route="local",
        ground_truth="15, 8, 50000",
        context=["The project has 15 members, completed 8 tasks, and has a budget of $50,000."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract all email addresses from the document",
        query_type="extract",
        expected_route="local",
        ground_truth="john@example.com, support@company.org",
        context=["Contact john@example.com for questions or support@company.org for help."],
        difficulty="medium"
    ),
    TestQuery(
        query="List all dates mentioned in the text",
        query_type="extract",
        expected_route="local",
        ground_truth="January 15, 2024; March 3, 2024; December 25, 2023",
        context=["The meeting is on January 15, 2024. The deadline was March 3, 2024. The project started on December 25, 2023."],
        difficulty="medium"
    ),
    TestQuery(
        query="Identify the colors mentioned in the description",
        query_type="extract",
        expected_route="local",
        ground_truth="Red, blue, green",
        context=["The flag has red stripes, a blue rectangle, and green stars."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract all phone numbers",
        query_type="extract",
        expected_route="local",
        ground_truth="555-1234, 555-5678",
        context=["Call us at 555-1234 or fax 555-5678 for assistance."],
        difficulty="medium"
    ),
    TestQuery(
        query="List the ingredients required",
        query_type="extract",
        expected_route="local",
        ground_truth="Flour, sugar, eggs, milk",
        context=["To make the cake, you will need flour, sugar, eggs, and milk."],
        difficulty="medium"
    ),
    TestQuery(
        query="What cities are listed as destinations",
        query_type="extract",
        expected_route="local",
        ground_truth="London, Paris, Tokyo",
        context=["The tour visits London, then moves to Paris, and ends in Tokyo."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract the dates of birth",
        query_type="extract",
        expected_route="local",
        ground_truth="1990-05-12, 1985-11-23",
        context=["Alice was born on 1990-05-12 and Bob on 1985-11-23."],
        difficulty="medium"
    ),
    TestQuery(
        query="Identify the programming languages mentioned",
        query_type="extract",
        expected_route="local",
        ground_truth="Python, Java, C++",
        context=["The project uses Python for backend, Java for android, and C++ for performance."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract the temperature readings",
        query_type="extract",
        expected_route="local",
        ground_truth="22°C, 24°C, 20°C",
        context=["Readings were 22°C in the morning, 24°C at noon, and 20°C at night."],
        difficulty="medium"
    ),
    TestQuery(
        query="List the stock tickers mentioned",
        query_type="extract",
        expected_route="local",
        ground_truth="AAPL, GOOGL, MSFT",
        context=["Tech giants like AAPL, GOOGL, and MSFT reported earnings."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract the authors of the paper",
        query_type="extract",
        expected_route="local",
        ground_truth="Smith, Johnson, Williams",
        context=["The study was conducted by Smith, Johnson, and Williams (2023)."],
        difficulty="medium"
    ),
    TestQuery(
        query="What are the dimensions of the box",
        query_type="extract",
        expected_route="local",
        ground_truth="10cm x 20cm x 30cm",
        context=["The package measures 10cm x 20cm x 30cm."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract the winning numbers",
        query_type="extract",
        expected_route="local",
        ground_truth="5, 12, 19, 24",
        context=["The winning numbers are 5, 12, 19, and 24."],
        difficulty="medium"
    ),
    TestQuery(
        query="Extract the names of the planets",
        query_type="extract",
        expected_route="local",
        ground_truth="Mars, Venus, Jupiter",
        context=["The mission will fly by Mars, Venus, and possibly Jupiter."],
        difficulty="medium"
    ),
    TestQuery(
        query="Identify the error codes",
        query_type="extract",
        expected_route="local",
        ground_truth="404, 500",
        context=["Server returned 404 for the first request and 500 for the second."],
        difficulty="medium"
    ),
]

# Multi-hop reasoning queries - should route to remote
MULTIHOP_QUERIES = [
    TestQuery(
        query="Analyze the economic implications of widespread AI adoption and recommend policy interventions",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Analysis should cover: job displacement concerns, productivity gains, wealth inequality, need for education reform, social safety net adjustments, and AI governance frameworks.",
        context=[
            "AI automation is expected to displace 85 million jobs by 2025 according to the World Economic Forum.",
            "McKinsey estimates AI could contribute $13 trillion to global GDP by 2030 through productivity gains.",
            "Studies show AI adoption may exacerbate wealth inequality as benefits accrue to capital owners and high-skilled workers.",
            "Education systems need reform to focus on skills that complement AI: creativity, emotional intelligence, complex problem-solving.",
            "Universal Basic Income (UBI) is proposed as a policy to address job displacement from automation.",
            "The EU AI Act represents comprehensive regulation covering high-risk AI applications and safety requirements.",
            "AI governance frameworks need to balance innovation with ethical considerations, privacy, and accountability."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Compare the advantages and disadvantages of renewable vs fossil fuel energy sources",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Comparison should include: cost, environmental impact, reliability, scalability, infrastructure requirements, and geopolitical implications.",
        context=[
            "Renewable energy sources (solar, wind, hydro) produce no direct greenhouse gas emissions during operation.",
            "Fossil fuels (coal, oil, gas) are reliable and can provide baseload power but emit significant CO2.",
            "Solar and wind costs have dropped 89% and 70% respectively since 2010, now cheaper than coal in many regions.",
            "Renewable energy is intermittent - solar doesn't work at night, wind varies with weather patterns.",
            "Fossil fuel infrastructure is already established globally, while renewables require massive new investments.",
            "Battery storage technology is improving but remains expensive for grid-scale deployment.",
            "Energy transitions affect geopolitics - oil-dependent economies face risks as world shifts to renewables."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Why did the Roman Empire fall",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Multiple interconnected factors including: military overextension, economic troubles, political instability, barbarian invasions, and administrative challenges.",
        context=[
            "The Roman Empire overextended militarily, with borders stretching from Britain to Mesopotamia by 117 AD.",
            "Economic decline included currency debasement, inflation, and heavy taxation to fund the military.",
            "Political instability: 50 emperors in 50 years during the Crisis of the Third Century (235-284 AD).",
            "Barbarian invasions increased from Germanic tribes, Huns, and Goths pressuring borders.",
            "The empire split into Western and Eastern halves in 285 AD, weakening unified defense.",
            "Administrative challenges: corruption, inefficient bureaucracy, and difficulty governing vast territories.",
            "The Western Roman Empire officially fell in 476 AD when Odoacer deposed Emperor Romulus Augustulus."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Evaluate the impact of social media on democratic processes and suggest improvements",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should cover misinformation spread, echo chambers, political polarization, foreign interference, and regulatory solutions.",
        context=[
            "Social media platforms amplify misinformation - false stories spread 6x faster than true ones on Twitter.",
            "Echo chambers form as algorithms show users content matching their existing beliefs.",
            "Political polarization has increased in countries with high social media usage.",
            "Foreign actors use social media to interfere in elections through targeted disinformation campaigns.",
            "Platform moderation policies struggle to balance free speech with harmful content removal.",
            "Potential solutions include transparency in algorithms, digital literacy education, and fact-checking integration."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Discuss the ethical considerations of gene editing technology like CRISPR",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should address: medical benefits, designer babies concerns, equity of access, unintended consequences, and need for regulation.",
        context=[
            "CRISPR allows precise editing of genes to treat genetic diseases like sickle cell anemia.",
            "Concerns exist about 'designer babies' - selecting traits for enhancement rather than treating disease.",
            "Gene editing technology is expensive, raising equity concerns about who can access benefits.",
            "Off-target effects and unintended mutations are risks that need more research.",
            "Germline editing (heritable changes) is more controversial than somatic editing (non-heritable).",
            "International scientific community calls for moratorium on human germline editing until safety established."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="How do supply chain disruptions affect global inflation and what can be done",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should explain: supply-demand imbalance, transportation costs, consumer prices, and policy responses.",
        context=[
            "COVID-19 disrupted global supply chains causing shortages of goods from semiconductors to consumer products.",
            "Container shipping costs increased 10x from 2020 to 2021 peak.",
            "When supply decreases but demand remains high, prices rise causing inflation.",
            "Just-in-time inventory systems are efficient but fragile during disruptions.",
            "Solutions include: diversifying suppliers, reshoring production, building inventory buffers.",
            "Central banks face dilemma: raise rates to fight inflation but risk slowing economic recovery."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="What are the implications of quantum computing for cybersecurity",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should cover: breaking current encryption, post-quantum cryptography, timeline concerns, and preparation strategies.",
        context=[
            "Quantum computers can break widely-used RSA and ECC encryption using Shor's algorithm.",
            "Current secure communications could become vulnerable when quantum computers scale up.",
            "Post-quantum cryptography algorithms are being developed to resist quantum attacks.",
            "NIST is standardizing post-quantum cryptographic algorithms for future use.",
            "Timeline uncertain - practical quantum computers may be 10-20 years away.",
            "'Harvest now, decrypt later' attacks: adversaries store encrypted data to decrypt when quantum computers available."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Analyze the relationship between urbanization and climate change",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should discuss: urban heat islands, emissions from cities, vulnerability to climate impacts, and sustainable urban planning.",
        context=[
            "Cities account for 75% of global CO2 emissions despite covering only 2% of Earth's surface.",
            "Urban heat island effect makes cities 1-7°F warmer than surrounding areas.",
            "Dense urban areas are more vulnerable to climate impacts like flooding and extreme heat.",
            "Sustainable urban planning: public transit, green spaces, energy-efficient buildings reduce emissions.",
            "By 2050, 68% of world's population will live in urban areas.",
            "Cities also offer opportunities for efficiency - compact living reduces per capita emissions."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Explain the causes and consequences of the French Revolution",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Causes: social inequality, economic crisis, Enlightenment ideas. Consequences: end of monarchy, rise of Napoleon, spread of democratic ideals.",
        context=[
            "The French Revolution (1789-1799) was driven by social inequality between the Three Estates.",
            "France faced a severe financial crisis due to war debts and poor harvests.",
            "Enlightenment thinkers like Rousseau and Voltaire challenged the divine right of kings.",
            "The revolution led to the execution of Louis XVI and the Reign of Terror.",
            "It eventually resulted in the rise of Napoleon Bonaparte and the Napoleonic Wars.",
            "Long-term effects included the spread of nationalism and liberalism across Europe."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Compare and contrast mitosis and meiosis",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Comparison should include: purpose (growth vs reproduction), number of divisions, daughter cells (2 diploid vs 4 haploid), and genetic variation.",
        context=[
            "Mitosis results in two genetically identical daughter cells.",
            "Meiosis produces four genetically distinct haploid gametes.",
            "Mitosis is used for growth and tissue repair.",
            "Meiosis is used for sexual reproduction.",
            "Crossing over occurs in Prophase I of meiosis, increasing genetic diversity.",
            "Mitosis involves one division; meiosis involves two successive divisions."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Discuss the impact of the printing press on European society",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should cover: increased literacy, spread of knowledge, Reformation, scientific revolution, and standardization of language.",
        context=[
            "Gutenberg's printing press (c. 1440) allowed for mass production of books.",
            "It drastically reduced the cost of books, increasing literacy rates.",
            "The spread of ideas facilitated the Protestant Reformation by disseminating Luther's theses.",
            "It enabled the rapid sharing of scientific discoveries, fueling the Scientific Revolution.",
            "Printing helped standardize vernicular languages and grammar."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Evaluate the pros and cons of nuclear energy",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Pros: low carbon emissions, high energy density, reliability. Cons: radioactive waste, accident risk, high initial cost.",
        context=[
            "Nuclear energy generates power through fission, producing zero carbon emissions during operation.",
            "It provides a reliable baseload power source unlike intermittent renewables.",
            "High energy density means a small amount of fuel produces vast energy.",
            "Radioactive waste disposal remains a long-term environmental challenge.",
            "Accidents like Chernobyl and Fukushima highlight safety risks.",
            "Building nuclear plants is capital-intensive and time-consuming."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="How does monetary policy influence inflation and unemployment",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should explain: interest rates, money supply, the Phillips curve trade-off.",
        context=[
            "Central banks use interest rates to control money supply.",
            "Higher interest rates reduce spending and inflation but can increase unemployment.",
            "Lower interest rates stimulate the economy and reduce unemployment but can cause inflation.",
            "The Phillips curve suggests an inverse relationship between inflation and unemployment in the short run.",
            "Quantitative easing is a tool to increase money supply during recessions."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Analyze the themes in 'To Kill a Mockingbird'",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Themes: racial injustice, loss of innocence, moral education, empathy, and courage.",
        context=[
            "The novel explores racial injustice in the American South through the trial of Tom Robinson.",
            "Scout and Jem's loss of innocence is central as they witness prejudice.",
            "Atticus Finch represents moral courage and integrity.",
            "The mockingbird symbolizes innocence effectively destroyed by evil.",
            "Empathy is taught through Atticus's advice to 'climb into someone's skin and walk around in it'."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="What are the physiological effects of stress on the human body",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should cover: cortisol release, cardiovascular impact, immune system suppression, and mental health.",
        context=[
            "Stress triggers the 'fight or flight' response, releasing cortisol and adrenaline.",
            "Chronic stress can lead to high blood pressure and heart disease.",
            "It suppresses the immune system, making the body more susceptible to infections.",
            "Long-term stress is linked to anxiety, depression, and sleep disorders.",
            "Stress affects digestion and can cause headaches or muscle tension."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Explain the concept of 'opportunity cost' with examples",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Definition: the value of the next best alternative foregone. Examples: staying in school vs working, buying a car vs investing.",
        context=[
            "Opportunity cost is a fundamental economic concept representing potential benefits an individual, investor, or business misses out on when choosing one alternative over another.",
            "It helps in making informed decisions by considering what is being given up.",
            "Example: If you spend an hour studying, the opportunity cost is the hour of leisure you could have had.",
            "It applies to resource allocation in production possibilities."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Describe the water cycle and its importance",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Stages: evaporation, condensation, precipitation, collection. Importance: fresh water distribution, climate regulation.",
        context=[
            "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.",
            "Evaporation turns liquid water into vapor; transpiration releases water from plants.",
            "Condensation forms clouds; precipitation returns water to Earth as rain or snow.",
            "Runoff collects in bodies of water.",
            "It is crucial for sustaining life, regulating temperature, and shaping landscapes."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="What were the main causes of World War I",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="M.A.I.N.: Militarism, Alliances, Imperialism, Nationalism, plus the assassination of Archduke Franz Ferdinand.",
        context=[
            "Militarism: Arms race between major powers like Britain and Germany.",
            "Alliances: Complex web of treaties (Triple Entente vs Triple Alliance) pulled nations into war.",
            "Imperialism: Competition for colonies and resources increased tension.",
            "Nationalism: Desire for self-determination, especially in the Balkans.",
            "The assassination of Archduke Franz Ferdinand in Sarajevo was the immediate trigger."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Compare classical and operant conditioning",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Classical: association between involuntary response and stimulus (Pavlov). Operant: association between voluntary behavior and consequence (Skinner).",
        context=[
            "Classical conditioning (Pavlov) involves learning through association of stimuli.",
            "It deals with involuntary, reflexive responses.",
            "Operant conditioning (Skinner) involves learning through rewards and punishments.",
            "It modifies voluntary behaviors.",
            "Both are forms of associative learning in behavioral psychology."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="How does the greenhouse effect work",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Process: solar energy reaches Earth, some reflected, some absorbed and re-radiated as heat, greenhouse gases trap this heat.",
        context=[
            "The greenhouse effect is a natural process that warms the Earth's surface.",
            "Sun's energy reaches Earth's atmosphere; some is reflected back to space.",
            "The rest is absorbed and re-radiated by greenhouse gases (CO2, methane, water vapor).",
            "This trapped heat maintains Earth's temperature at a habitable level.",
            "Human activities have intensified this effect, leading to global warming."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Discuss the impact of the internet on globalization",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Should cover: communication speed, global commerce, cultural exchange, remote work, and digital divide.",
        context=[
            "The internet has drastically lowered communication costs and increased speed globally.",
            "It enables e-commerce and global supply chains.",
            "Cultural exchange is facilitated through social media and content streaming.",
            "It allows for the outsourcing of services and rise of remote work.",
            "However, it has created a 'digital divide' between connected and unconnected regions."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Analyze the role of enzymes in biological reactions",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Role: biological catalysts, lower activation energy, specific to substrates, affected by pH and temperature.",
        context=[
            "Enzymes are proteins that act as biological catalysts.",
            "They speed up chemical reactions by lowering the activation energy.",
            "Each enzyme is specific to a substrate (lock and key model).",
            "Enzyme activity is influenced by factors like temperature, pH, and concentration.",
            "They are essential for digestion, metabolism, and DNA replication."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="How does vaccine-induced immunity work",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Process: vaccine introduces antigen, immune system produces antibodies and memory cells, faster response upon reinfection.",
        context=[
            "Vaccines stimulate the body's adaptive immunity.",
            "They introduce a weakened or inactive part of a pathogen (antigen).",
            "The immune system recognizes it as foreign and produces antibodies.",
            "Memory B and T cells are formed, providing long-term protection.",
            "Herd immunity occurs when a large portion of the population is vaccinated."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Discuss the ethical implications of AI surveillance",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Privacy, bias, lack of consent, potential for abuse, chilling effect on free speech.",
        context=[
             "AI surveillance enables mass monitoring of populations.",
             "Facial recognition technology can be biased against certain demographics.",
             "Privacy advocates argue it infringes on civil liberties.",
             "Governments justify it for national security and crime prevention."
        ],
        difficulty="hard"
    ),
    TestQuery(
        query="Compare the healthcare systems of the US and Canada",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="US: mixed public/private, higher cost, variable access. Canada: single-payer publicly funded, lower cost, universal coverage, potential wait times.",
        context=[
             "The US healthcare system is a mix of private insurance and public programs like Medicare.",
             "Canada has a publicly funded, single-payer health care system.",
             "The US spends more per capita on healthcare than any other nation.",
             "Canadian citizens have universal coverage for medically necessary services."
        ],
        difficulty="hard"
    ),
]

# Code queries - moderate
CODE_QUERIES = [
    TestQuery(
        query="Write a Python function to check if a number is prime",
        query_type="code",
        expected_route="local",
        ground_truth="Should include: checking if n <= 1, iterating up to sqrt(n), checking divisibility",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a function to reverse a string",
        query_type="code",
        expected_route="local",
        ground_truth="Should return reversed string, e.g., 'hello' becomes 'olleh'",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Create a function to find the maximum element in a list",
        query_type="code",
        expected_route="local",
        ground_truth="Should handle empty lists and return the largest value",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Write a Python function to calculate factorial of a number",
        query_type="code",
        expected_route="local",
        ground_truth="Should handle base case n=0 or n=1, use recursion or iteration",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a function to calculate the Fibonacci sequence up to n",
        query_type="code",
        expected_route="remote",
        ground_truth="Should include: base cases, iterative or recursive logic",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Create a class for a Bank Account with deposit and withdraw methods",
        query_type="code",
        expected_route="remote",
        ground_truth="Class structure, balance attribute, methods with logic checks",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a SQL query to select all users over 18",
        query_type="code",
        expected_route="local",
        ground_truth="SELECT * FROM users WHERE age > 18;",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Implement a bubble sort algorithm",
        query_type="code",
        expected_route="remote",
        ground_truth="Nested loops, swapping logic",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a function to check if a string is a palindrome",
        query_type="code",
        expected_route="local",
        ground_truth="Check string against its reverse, ignore case/spaces if needed",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Create a dictionary comprehension to square numbers from 1 to 5",
        query_type="code",
        expected_route="local",
        ground_truth="{x: x**2 for x in range(1, 6)}",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Write a regular expression to validate an email address",
        query_type="code",
        expected_route="remote",
        ground_truth="Regex pattern matching user, domain, extension",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Implement a function to merge two sorted lists",
        query_type="code",
        expected_route="remote",
        ground_truth="Iterate through both lists, comparing elements, appending smaller",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a script to read a JSON file and print a specific key",
        query_type="code",
        expected_route="remote",
        ground_truth="Import json, open file, json.load, print key",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Create a simple decorator to time a function execution",
        query_type="code",
        expected_route="remote",
        ground_truth="Wrapper function, time.time, call func, print diff",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a function to flatten a nested list",
        query_type="code",
        expected_route="remote",
        ground_truth="Recursive or iterative approach to process sub-lists",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a Python script to scrape a webpage using BeautifulSoup",
        query_type="code",
        expected_route="remote",
        ground_truth="Import requests/BS4, get URL, parse content, find elements",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Create a React component for a counter",
        query_type="code",
        expected_route="remote",
        ground_truth="Function component, useState hook, increment button",
        context=[],
        difficulty="medium"
    ),
]

# Open-ended queries - prefer remote
OPENENDED_QUERIES = [
    TestQuery(
        query="What should I do this weekend",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Depends on personal preferences, should offer varied suggestions",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="How can I improve my productivity at work",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Should include time management, focus techniques, organization strategies",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Suggest a healthy meal plan for a week",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Balanced diet including proteins, veggies, carbs",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a short story about a time traveler",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Creative narrative, plot structure, characters",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="What are some creative gift ideas for a gardener",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Tools, seeds, plants, decor, books",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Propose a marketing strategy for a new coffee shop",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Social media, loyalty program, local events, partnerships",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Explain the plot of the movie Inception to a 5-year-old",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Simplified explanation about dreams within dreams",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Draft a resignation letter",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Professional tone, effective date, gratitude",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Suggest a workout routine for beginners",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Mix of cardio and strength, warm-up/cool-down, manageable intensity",
        context=[],
        difficulty="medium"
    ),
    TestQuery(
        query="Write a poem about the ocean",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Creative writing about waves, depth, blue color, marine life",
        context=[],
        difficulty="medium"
    ),
]


class TestDataset:
    """Container for all test queries"""

    def __init__(self):
        self.all_queries = (
            LOOKUP_QUERIES +
            MATH_QUERIES +
            EXTRACT_QUERIES +
            MULTIHOP_QUERIES +
            CODE_QUERIES +
            OPENENDED_QUERIES
        )

        # Group by type
        self.queries_by_type = {
            "lookup": LOOKUP_QUERIES,
            "math": MATH_QUERIES,
            "extract": EXTRACT_QUERIES,
            "multi-hop": MULTIHOP_QUERIES,
            "code": CODE_QUERIES,
            "open-ended": OPENENDED_QUERIES,
        }

        # Group by expected route
        self.queries_by_route = {
            "local": [q for q in self.all_queries if q.expected_route == "local"],
            "remote": [q for q in self.all_queries if q.expected_route == "remote"],
        }

        # Group by difficulty
        self.queries_by_difficulty = {
            "easy": [q for q in self.all_queries if q.difficulty == "easy"],
            "medium": [q for q in self.all_queries if q.difficulty == "medium"],
            "hard": [q for q in self.all_queries if q.difficulty == "hard"],
        }

    def get_all(self) -> List[TestQuery]:
        """Get all test queries"""
        return self.all_queries

    def get_by_type(self, query_type: str) -> List[TestQuery]:
        """Get queries of a specific type"""
        return self.queries_by_type.get(query_type, [])

    def get_by_route(self, expected_route: str) -> List[TestQuery]:
        """Get queries expected to route a certain way"""
        return self.queries_by_route.get(expected_route, [])

    def get_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """Get queries of a specific difficulty"""
        return self.queries_by_difficulty.get(difficulty, [])

    def summary(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            "total_queries": len(self.all_queries),
            "by_type": {k: len(v) for k, v in self.queries_by_type.items()},
            "by_route": {k: len(v) for k, v in self.queries_by_route.items()},
            "by_difficulty": {k: len(v) for k, v in self.queries_by_difficulty.items()},
        }


if __name__ == "__main__":
    # Demo
    dataset = TestDataset()
    print("Test Dataset Summary")
    print("=" * 60)

    summary = dataset.summary()
    print(f"Total queries: {summary['total_queries']}")
    print(f"\nBy type: {summary['by_type']}")
    print(f"By expected route: {summary['by_route']}")
    print(f"By difficulty: {summary['by_difficulty']}")

    print("\nSample queries:")
    for query_type in ["lookup", "math", "multi-hop"]:
        queries = dataset.get_by_type(query_type)
        if queries:
            print(f"\n{query_type.upper()}: {queries[0].query}")
