"""
Synthetic Test Dataset for Smart Router Experiments

This module provides test queries categorized by type with expected behaviors
and ground truth answers for evaluation.
"""

from typing import List, Dict, Any
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
        query="Extract all product names and their prices",
        query_type="extract",
        expected_route="local",
        ground_truth="Laptop - $999, Mouse - $25, Keyboard - $75",
        context=["We offer: Laptop for $999, Mouse for $25, and Keyboard for $75."],
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
        query="Implement a binary search algorithm in Python",
        query_type="code",
        expected_route="remote",
        ground_truth="Should include: sorted array check, mid calculation, recursive/iterative approach",
        context=[],
        difficulty="hard"
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
        query="What are some good books to read for personal development",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Should suggest diverse books across different development areas",
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
