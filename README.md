# LangChain RAG Implementation - Coriander Knowledge Base

A comprehensive RAG (Retrieval-Augmented Generation) system built with LangChain, specifically designed for coriander (cilantro) knowledge extraction and Q&A. This system scrapes coriander information from Wikipedia, creates a vector database, and provides an interactive Q&A system with source attribution.

## 🌟 Key Features

- 🔍 **Web Scraping**: Automatically scrapes coriander information from Wikipedia
- 📝 **Document Processing**: Intelligent document splitting and preprocessing
- 🧠 **Vectorization**: Uses M3E-base model for text embedding
- 💾 **Vector Storage**: Chroma database for efficient vector storage
- 🤖 **Smart Q&A**: Retrieval-augmented generation with source attribution
- 💬 **Interactive Dialogue**: Multi-turn conversations with context memory
- 📊 **Confidence Scoring**: Source relevance and confidence metrics
- 🔍 **Source Attribution**: Detailed source document analysis

## 🚀 Quick Start

### One-Click Setup (Recommended)

```bash
# Complete setup (environment + model + test)
./00_setup_env.sh
./01_download_m3e.sh
./02_run_rag.sh
```

### Step-by-Step Setup

#### Step 1: Environment Setup
```bash
./00_setup_env.sh
```

#### Step 2: Download Model
```bash
./01_download_m3e.sh
```

#### Step 3: Run System
```bash
./02_run_rag.sh
```

## ⚙️ API Configuration

Before running the system, configure your DeepSeek API key:

### Method 1: Using .env file (Recommended)

1. Create a `.env` file in the project root:
   ```bash
   echo "DEEPSEEK_API_KEY=your_deepseek_api_key_here" > .env
   ```

2. Replace `your_deepseek_api_key_here` with your actual API key.

### Method 2: Direct Code Modification

Edit `rag_implementation.py` and find this line:
```python
api_key=os.getenv("DEEPSEEK_API_KEY"),
```

Replace with:
```python
api_key="your_deepseek_api_key_here",
```

### Get API Key

- DeepSeek API: [DeepSeek Console](https://platform.deepseek.com/)




## 🏗️ Technical Architecture

### Core Components

- **Web Scraping**: `requests` + `BeautifulSoup` for Wikipedia extraction
- **Document Processing**: `TextLoader` + `RecursiveCharacterTextSplitter`
- **Vectorization**: `HuggingFaceBgeEmbeddings` (M3E-base model)
- **Vector Storage**: `Chroma` database
- **LLM Integration**: `ChatOpenAI` with DeepSeek API
- **Conversation Chain**: `ConversationalRetrievalChain`



## 📁 Project Structure

```
12_Langchain_RAG/
├── rag_implementation.py    # Main implementation file
├── requirements.txt         # Python dependencies
├── README.md              # This documentation
├── 00_setup_env.sh        # Environment setup script
├── 01_download_m3e.sh     # Model download script
├── 02_run_rag.sh          # Main execution script
├── coriander.txt          # Scraped coriander data
├── AI-ModelScope/         # Local model storage
    └── m3e-base/         # M3E embedding model
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- 2GB+ free disk space for model download
- Internet connection for API access

### Automated Installation

```bash
# Clone the repository
git clone <repository-url>
cd 12_Langchain_RAG

# Run complete setup
./00_setup_env.sh
./01_download_m3e.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv rag_env

# Activate environment
source rag_env/bin/activate  # macOS/Linux
# or
rag_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download model
./01_download_m3e.sh
```

## 🚀 Usage

### Running the System

```bash
# Activate environment
source rag_env/bin/activate

# Run the RAG system
python rag_implementation.py
```

### System Workflow

1. **Data Scraping**: Automatically scrapes coriander information from Wikipedia
2. **Document Processing**: Splits content into vectorizable chunks
3. **Vectorization**: Converts text to vectors using M3E model
4. **Storage**: Stores vectors in Chroma database
5. **Q&A Interface**: Launches interactive question-answering system

### Interactive Commands

- Type your questions about coriander
- Type `quit` or `exit` to exit the system
- Type `sources` to see detailed source information
- Answer `y` when prompted to view full source documents

## 📈 Performance Features

### Document Processing
- **Chunk Size**: 128 tokens with 50 token overlap
- **Split Strategy**: Recursive character splitting with optimized separators
- **Processing Speed**: Handles large documents efficiently

### Retrieval System
- **Search Type**: Similarity-based retrieval
- **Top-k**: Retrieves top 5 most relevant documents
- **Confidence Scoring**: Relevance and length-based scoring

### Memory Management
- **Conversation Buffer**: Maintains chat history
- **Context Preservation**: Multi-turn conversation support
- **Memory Optimization**: Efficient memory usage

## 🎯 Use Cases

### Primary Use Case: Coriander Knowledge Q&A

The system is specifically designed for coriander (cilantro) knowledge extraction and question answering. Here are the main use cases:

#### 1. **Agricultural Knowledge Base**
- **Planting Instructions**: How to plant, grow, and harvest coriander
- **Growing Conditions**: Soil requirements, climate needs, watering schedules
- **Pest Management**: Common diseases and pest control methods
- **Harvesting Techniques**: When and how to harvest leaves and seeds

#### 2. **Culinary Applications**
- **Cooking Methods**: How to use coriander in different cuisines
- **Flavor Profiles**: Taste characteristics and pairing suggestions
- **Storage Techniques**: How to preserve coriander leaves and seeds
- **Nutritional Information**: Health benefits and nutritional content

#### 3. **Research and Education**
- **Academic Research**: Detailed botanical and agricultural information
- **Educational Content**: Structured learning about coriander
- **Comparative Analysis**: Different varieties and their characteristics

### Example Questions You Can Ask:

#### Growing Coriander:
- "How to plant coriander?"
- "What are the best growing conditions for coriander?"
- "When should I harvest coriander leaves?"
- "How to prevent coriander from bolting?"

#### Culinary Uses:
- "How to use coriander in cooking?"
- "What are the health benefits of coriander?"
- "How to store coriander seeds?"
- "What's the difference between coriander and cilantro?"

#### Technical Information:
- "What is the scientific name of coriander?"
- "What are the nutritional properties of coriander seeds?"
- "How is coriander used in traditional medicine?"



```text
Example:

Please enter your question: How to use coriander in cooking?

==================================================
Answer: Coriander can be used in cooking in different ways depending on whether you're using the **seeds** (a spice) or the **fresh leaves** (also called cilantro in some regions). Here’s how to use both:

### **Coriander Seeds (Spice)**
- **Whole Seeds**: Often used in pickling, brines, or slow-cooked dishes like stews and curries. They release flavor when toasted or simmered.
- **Ground Coriander**: Best freshly ground for maximum flavor. Used in spice blends (like garam masala), marinades, soups, and roasted meats.
- **Toasting**: Lightly dry-toast seeds before grinding to enhance their nutty, citrusy flavor.
- **Storage**: Whole seeds last longer; ground coriander loses flavor quickly, so grind as needed.

### **Fresh Coriander Leaves (Cilantro)**
- **Raw Use**: Best added at the end of cooking or as a garnish (heat diminishes its flavor). Great in salsas, salads, chutneys, and Vietnamese pho.
- **Stems**: Finely chopped stems can be used in marinades, dressings, or blended into sauces (they have strong flavor).
- **Herb Blends**: Often paired with mint, parsley, or lime in dishes like tacos, curries, and Thai cuisine.

### **Examples of Dishes Using Coriander:**
- **Seeds**: Curries, soups, pickles, spice rubs, and baked goods.
- **Leaves**: Guacamole, salsa verde, chutneys, noodle dishes, and garnishes for soups.

Would you like specific recipes or pairings?
==================================================

📚 Source Attribution:
Based on 5 relevant document chunks:
--------------------------------------------------

Source 5 (Confidence: 0.14):
Relevance Score: 0.12
Content: Coriander is listed as one of the original ingredients in thesecret formulaforCoca-Cola.[48]
Metadata: {'source': './coriander.txt'}
------------------------------

Source 1 (Confidence: 0.14):
Relevance Score: 0.12
Content: Coriander seeds are one of the key botanicals used to flavorgin.[citation needed]
Metadata: {'source': './coriander.txt'}
------------------------------

Source 4 (Confidence: 0.13):
Relevance Score: 0.11
Content: . The coriander seeds are used with orange peel to add a citrus character.[citation needed]
Metadata: {'source': './coriander.txt'}
------------------------------

Source 2 (Confidence: 0.09):
Relevance Score: 0.06
Content: Coriander is commonly found both as whole dried seeds and ingroundform
Metadata: {'source': './coriander.txt'}
------------------------------

Source 3 (Confidence: 0.06):
Relevance Score: 0.00
Content: .[25]TheEbers Papyrus, an Egyptian text dated around 1550BCE, mentioned uses of coriander.[32]
Metadata: {'source': './coriander.txt'}
------------------------------

Would you like to see the full source documents? (y/n): .rag_env/bin/activate

Please enter your question: exit
Thank you for using!
RAG system exited

```