# **Scene Graph Generator**

A React-based application for generating and visualizing scene graphs using D3.js. This tool allows users to input prompts and dynamically render scene graphs with nodes, edges, and directional arrows, representing relationships between objects, attributes, and actions.

---

## **Features**

- **Dynamic Scene Graph Generation**:
  - Nodes are categorized into three types: `objects`, `attributes`, and `relationships`.
  - Nodes are color-coded for better visual distinction:
    - **Red**: Objects
    - **Blue**: Attributes
    - **Green**: Relationships

- **Interactive Graph Visualization**:
  - Drag nodes to reposition them dynamically.
  - Force-directed layout ensures an aesthetically pleasing graph structure.
  - Links (edges) between nodes are directional, with arrows representing relationships.

- **Customizable Design**:
  - Adjustable node size based on text length.
  - Rounded node corners for a polished look.
  - Clear and minimalistic arrow styles for better readability.

---

## **Installation**

### 1. **Clone the Repository**:

```bash
git clone https://github.com/your-repository/scene-graph-generator.git
cd scene-graph-generator
```

### 2. **Install Dependencies**:

If you are using Yarn:
```bash
yarn install
```
If you are using npm:
```bash
npm install
```

### 3. **Server**

You should use the server to generate image using model. 
Clone the LAION-SG repository, and follow the steps in the repository.


*Warning: GPU should be available*

```bash
git clone https://github.com/mengcye/LAION-SG
cd LAION-SG
pip install -r requirements.txt
# follow steps in the repository
```


### 4. **Start the Application**:

If you are using Yarn:
```bash
yarn start
```
If you are using npm:
```bash
npm start
```

For the server, open a new terminal and run the following:
```bash
cd LAION-SG
python server.py
```
---

## **Usage**

1. **Input Prompt**:
   Enter a descriptive text in the input field to define the scene. For example:
   ```
   A brown cat sitting on a wooden bench holding an ice cream.
   ```

2. **Generate Scene Graph**:
   Click the "Generate" button to create the scene graph.

3. **Interact with the Graph**:
   - Drag nodes to explore the graph.
   - Observe the relationships represented by directional arrows.

---

## **Technologies Used**

- **React.js**:
  - For building the interactive user interface.
- **D3.js**:
  - For graph visualization and rendering force-directed layouts.

---

## **File Structure**

```plaintext
src/
├── components/
│   └── SceneGraph.jsx   # Main component for rendering the graph
├── App.js               # Root application component
├── index.js             # Entry point of the application
└── styles.css           # Global styles (if applicable)
```

---

## **Customization**

- **Node Colors**:
  Modify the `fill` attribute in the `SceneGraph.jsx` file to change node colors:
  ```javascript
  .attr("fill", (d) =>
    d.type === "object"
      ? "#ff6b6b" // Red for objects
      : d.type === "attribute"
      ? "#4dabf7" // Blue for attributes
      : "#51cf66" // Green for relationships
  );
  ```

- **Force Layout Settings**:
  Adjust the simulation forces to modify graph spacing:
  ```javascript
  .force("link", d3.forceLink().distance(150)) // Link distance
  .force("charge", d3.forceManyBody().strength(-300)) // Node repulsion
  ```

---

## **Future Improvements**

- Add the ability to save and export graphs as images or JSON files.
- Support for hierarchical layouts for more complex graphs.
- Integration with AI models to automatically parse prompts into scene graphs.

---

## **Contributing**

Contributions are welcome! If you'd like to improve the project or add new features:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit and push your changes:
   ```bash
   git commit -m "Add your message here"
   git push origin feature-name
   ```
4. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## **Acknowledgments**

- Built with ❤️ using **React** and **D3.js**.
- Inspired by the need for intuitive scene graph visualization tools.

Feel free to reach out if you have any questions or suggestions! 😊