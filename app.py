import os
import sqlite3
import random
import json
import openai
from datetime import timedelta
from faker import Faker
from langgraph.graph import StateGraph, START
from typing import TypedDict, Optional
import gradio as gr

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Initialize SQLite Database with Faker Data ------------------
def init_db():
    fake = Faker()
    conn = sqlite3.connect("complex_test_db.sqlite", timeout=20)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist (for a clean setup)
    cursor.executescript("""
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS payments;
        DROP TABLE IF EXISTS shipment;
    """)
    
    # Create Customers Table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            address TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create Products Table
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            stock_quantity INTEGER NOT NULL
        );
    """)
    
    # Create Orders Table
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount DECIMAL(10,2) NOT NULL,
            status TEXT CHECK(status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)
    
    # Create Order Items Table (Many-to-Many relationship between Orders and Products)
    cursor.execute("""
        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            subtotal DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
    """)
    
    # Create Payments Table (One-to-One relationship with Orders)
    cursor.execute("""
        CREATE TABLE payments (
            payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER UNIQUE NOT NULL,
            payment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            amount DECIMAL(10,2) NOT NULL,
            payment_method TEXT CHECK(payment_method IN ('Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer')) NOT NULL,
            status TEXT CHECK(status IN ('Success', 'Failed', 'Pending')) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """)
    
    # Create Shipment Table (One-to-One relationship with Orders)
    cursor.execute("""
        CREATE TABLE shipment (
            shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER UNIQUE NOT NULL,
            shipment_date TIMESTAMP,
            delivery_date TIMESTAMP,
            carrier TEXT NOT NULL,
            tracking_number TEXT UNIQUE,
            status TEXT CHECK(status IN ('Processing', 'Shipped', 'Delivered')) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """)
    
    conn.commit()
    
    # Insert substantial data
    NUM_CUSTOMERS = 1000
    NUM_PRODUCTS = 500
    NUM_ORDERS = 2000
    NUM_ORDER_ITEMS = 5000
    NUM_PAYMENTS = NUM_ORDERS
    NUM_SHIPMENTS = int(NUM_ORDERS * 0.8)  # 80% of orders are shipped
    
    # Insert Customers (Ensure unique emails)
    customers = []
    unique_emails = set()
    while len(customers) < NUM_CUSTOMERS:
        name = fake.name()
        email = fake.email()
        phone = fake.phone_number()
        address = fake.address().replace("\n", ", ")
        if email not in unique_emails:
            customers.append((name, email, phone, address))
            unique_emails.add(email)
    
    cursor.executemany("""
        INSERT INTO customers (name, email, phone, address)
        VALUES (?, ?, ?, ?);
    """, customers)
    
    # Insert Products
    products = []
    categories = ["Electronics", "Clothing", "Books", "Home Appliances", "Toys"]
    for _ in range(NUM_PRODUCTS):
        products.append((fake.word().capitalize(), random.choice(categories), round(random.uniform(5, 500), 2), random.randint(10, 500)))
    
    cursor.executemany("""
        INSERT INTO products (name, category, price, stock_quantity)
        VALUES (?, ?, ?, ?);
    """, products)
    
    # Fetch inserted customer and product IDs
    cursor.execute("SELECT customer_id FROM customers;")
    customer_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT product_id FROM products;")
    product_ids = [row[0] for row in cursor.fetchall()]
    
    # Insert Orders
    orders = []
    statuses = ["Pending", "Shipped", "Delivered", "Cancelled"]
    for _ in range(NUM_ORDERS):
        customer_id = random.choice(customer_ids)
        total_amount = round(random.uniform(20, 2000), 2)
        status = random.choice(statuses)
        order_date = fake.date_time_between(start_date="-1y", end_date="now")
        orders.append((customer_id, order_date, total_amount, status))
    
    cursor.executemany("""
        INSERT INTO orders (customer_id, order_date, total_amount, status)
        VALUES (?, ?, ?, ?);
    """, orders)
    
    # Fetch inserted order IDs
    cursor.execute("SELECT order_id FROM orders;")
    order_ids = [row[0] for row in cursor.fetchall()]
    
    # Insert Order Items
    order_items = []
    for _ in range(NUM_ORDER_ITEMS):
        order_id = random.choice(order_ids)
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 5)
        subtotal = round(quantity * random.uniform(5, 500), 2)
        order_items.append((order_id, product_id, quantity, subtotal))
    
    cursor.executemany("""
        INSERT INTO order_items (order_id, product_id, quantity, subtotal)
        VALUES (?, ?, ?, ?);
    """, order_items)
    
    # Insert Payments
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
    payment_statuses = ["Success", "Failed", "Pending"]
    payments = []
    for order_id in order_ids[:NUM_PAYMENTS]:
        amount = round(random.uniform(20, 2000), 2)
        payment_method = random.choice(payment_methods)
        status = random.choice(payment_statuses)
        payments.append((order_id, fake.date_time_between(start_date="-1y", end_date="now"), amount, payment_method, status))
    
    cursor.executemany("""
        INSERT INTO payments (order_id, payment_date, amount, payment_method, status)
        VALUES (?, ?, ?, ?, ?);
    """, payments)
    
    # Insert Shipments (for 80% of orders)
    carriers = ["FedEx", "UPS", "DHL", "USPS"]
    shipments = []
    for order_id in order_ids[:NUM_SHIPMENTS]:
        shipment_date = fake.date_time_between(start_date="-1y", end_date="now")
        delivery_date = shipment_date + timedelta(days=random.randint(1, 10))
        tracking_number = fake.uuid4()
        carrier = random.choice(carriers)
        status = random.choice(["Processing", "Shipped", "Delivered"])
        shipments.append((order_id, shipment_date, delivery_date, carrier, tracking_number, status))
    
    cursor.executemany("""
        INSERT INTO shipment (order_id, shipment_date, delivery_date, carrier, tracking_number, status)
        VALUES (?, ?, ?, ?, ?, ?);
    """, shipments)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database setup complete with complex relationships and substantial data!")

# Initialize the database on app startup
init_db()

# ------------------ Define State for the Workflow ------------------
class SQLExecutionState(TypedDict):
    sql_query: str
    structured_metadata: Optional[dict]
    validation_result: Optional[dict]
    optimized_sql: Optional[str]
    execution_result: Optional[dict]

# Initialize the LangGraph Workflow
graph = StateGraph(state_schema=SQLExecutionState)

# ------------------ 1. Query Understanding Agent ------------------
def query_understanding_agent(state: SQLExecutionState) -> SQLExecutionState:
    natural_language_query = state["sql_query"]
    prompt = f"""
    Convert the following natural language query into **structured SQL metadata** based on the database schema.
    If you cannot generate a query that adheres strictly to the schema, return:
    {{ "error": "Invalid query: Tables or columns do not match schema" }}

    **Query:** "{natural_language_query}"

    **Database Schema:**
    - **orders** (order_id, customer_id, order_date, total_amount, status)
    - **order_items** (order_item_id, order_id, product_id, quantity, subtotal)
    - **products** (product_id, name, category, price, stock_quantity)
    - **customers** (customer_id, name, email, phone, address, created_at)
    - **payments** (payment_id, order_id, payment_date, amount, payment_method, status)
    
    **Rules:**
    - Use only the provided tables.
    - Ensure correct column names.
    - Return output strictly in JSON format.
    - Group by relevant fields when necessary.

    **Example Output Format:**
    {json.dumps({
        "operation": "SELECT",
        "columns": ["customer_id", "SUM(total_amount) AS total_spent"],
        "table": "orders",
        "conditions": ["order_date BETWEEN '2024-01-01' AND '2024-12-31'"],
        "group_by": ["customer_id"],
        "order_by": ["total_spent DESC"],
        "limit": 5
    }, indent=4)}

    **DO NOT return explanations. Only return valid JSON.**
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        metadata = json.loads(response["choices"][0]["message"]["content"])
        return {"structured_metadata": metadata}
    except json.JSONDecodeError:
        return {"structured_metadata": {"error": "Invalid JSON response from OpenAI"}}

graph.add_node("Query Understanding", query_understanding_agent)

# ------------------ 2. Query Validation Agent ------------------
def query_validation_agent(state: SQLExecutionState) -> SQLExecutionState:
    sql_metadata = state.get("structured_metadata", {})
    if "error" in sql_metadata:
        return {"validation_result": {"error": sql_metadata["error"]}}
    query = sql_metadata.get("operation", "")
    restricted_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    if any(keyword in query.upper() for keyword in restricted_keywords):
        return {"validation_result": {"error": "Potentially harmful SQL operation detected!"}}
    return {"validation_result": {"valid": True}}

graph.add_node("Query Validation", query_validation_agent)

# ------------------ 3. Query Optimization Agent ------------------
def query_optimization_agent(state: SQLExecutionState) -> SQLExecutionState:
    sql_metadata = state.get("structured_metadata", {})
    prompt = f"""
    Optimize the following SQL query for performance while ensuring that the output includes only the required columns and necessary joins.
    Do not include any extra columns, unnecessary joins, or records that are not required to answer the query.

    Here is the original SQL metadata:
    {json.dumps(sql_metadata, indent=4)}

    Output only the final optimized SQL query in plain text without any markdown formatting or explanations.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    optimized_query = response["choices"][0]["message"]["content"].strip()
    if optimized_query.startswith("```sql"):
        optimized_query = optimized_query.replace("```sql", "").replace("```", "").strip()
    return {"optimized_sql": optimized_query}

graph.add_node("Query Optimization", query_optimization_agent)

# ------------------ 4. SQL Execution Agent ------------------
def execution_agent(state: SQLExecutionState) -> SQLExecutionState:
    query = state.get("optimized_sql", "").strip()
    if not query:
        return {"execution_result": {"error": "No SQL query to execute."}}
    try:
        conn = sqlite3.connect("complex_test_db.sqlite", timeout=20)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        if not result:
            return {"execution_result": {"error": "Query executed successfully but returned no results."}}
        return {"execution_result": result}
    except sqlite3.Error as e:
        return {"execution_result": {"error": str(e)}}

graph.add_node("SQL Execution", execution_agent)

# ------------------ Define Execution Flow ------------------
graph.add_edge(START, "Query Understanding")
graph.add_edge("Query Understanding", "Query Validation")
graph.add_edge("Query Validation", "Query Optimization")
graph.add_edge("Query Optimization", "SQL Execution")

compiled_pipeline = graph.compile()

# ------------------ Function to Run the Multi-Agent Query ------------------
def run_multi_agent_query(natural_language_query):
    result = compiled_pipeline.invoke({"sql_query": natural_language_query})
    return json.dumps(result.get("execution_result", {}), indent=2)

# ------------------ Gradio Interface ------------------
schema_description = """
**Database Schema:**

- **customers**: customer_id, name, email, phone, address, created_at  
- **products**: product_id, name, category, price, stock_quantity  
- **orders**: order_id, customer_id, order_date, total_amount, status  
- **order_items**: order_item_id, order_id, product_id, quantity, subtotal  
- **payments**: payment_id, order_id, payment_date, amount, payment_method, status  
- **shipment**: shipment_id, order_id, shipment_date, delivery_date, carrier, tracking_number, status  
"""

iface = gr.Interface(
    fn=run_multi_agent_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your natural language query here.(Which products sold the most in 2024))."),
    outputs="text",
    title="Multi-Agent SQL Generator",
    description=("Enter a natural language query to generate and execute a SQL query. E.g. "
                 "Find the email_id of the top 5 customers who spent the most in 2024.\n\n"
                 + schema_description)
)


if __name__ == "__main__":
    iface.launch()
