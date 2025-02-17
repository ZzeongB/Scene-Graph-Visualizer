
def scene_graph_to_triples(json_data):
    """
    {
    // sceneGraph is in the form of a JSON object, with {"objects":["attributes"], "relationships"}
    objects: [
      { id: "object1", name: "wolf" },
      { id: "object2", name: "icecream", attributes: ["chocolate"] },
    ],
    relationships: [
      { source: "object1", target: "object2", relation: "holding" },
    ],
  }
    
  
    """
    triples = []
    global_ids = []
    related_items = set()

    items = json_data["objects"]
    relations = json_data["relationships"]

    item_dict = {item["id"]: item for item in items}

    for relation in relations:
        item1_id = relation["source"]
        item2_id = relation["target"]

        related_items.add(item1_id)
        related_items.add(item2_id)

        # if attributes are present, add them to the item name
        if("attributes" in item_dict[item1_id]):
            item1_attributes = " ".join(item_dict[item1_id]["attributes"]) + " " + item_dict[item1_id]["name"]
        else:
            item1_attributes = item_dict[item1_id]["name"]
        if("attributes" in item_dict[item2_id]):
            item2_attributes = " ".join(item_dict[item2_id]["attributes"]) + " " + item_dict[item2_id]["name"]
        else:
            item2_attributes = item_dict[item2_id]["name"]

        triples.append({
            "item1": item1_attributes,
            "relation": relation["relation"],
            "item2": item2_attributes
        })

        global_ids.append({
            "item1": 0,
            "item2": 0,
        })


    isolated_items = []
    for item in items:
        if item["id"] not in related_items:
            if("attributes" in item):
                isolated_items.append(" ".join(item["attributes"]) + " " + item["name"])
            else:
                isolated_items.append(item["name"])

    return triples, global_ids, isolated_items

def generate_prompt(original_sg, new_sg, graph_changes):
    # Given the tracks of graph changes, generate series of ['mask', 'prompt'] for each change
    # Mask path is "attributes + object name" from original sg
    # Prompt is the new fill of the object name + attributes
    
    """
    { type: "add object", object: newNode.id, name: newNode.name };
    {
        type: "add attribute",
        object: objectId,
        attribute: attributeName,
        };
    {
        type: "add relationship",
        source: sourceId,
        target: targetId,
        relation: newEdge.relation,
    };
    { type: "rename object", object: node.id, name: newName };
    {
        type: "rename attribute",
        object: objectId,
        attribute: node.name,
        name: newName,
    };
    {
        type: "rename relationship",
        relation: node.name,
        name: newName,
    };
    { type: "delete object", object: node.id };
    {
        type: "delete attribute",
        object: objectId,å
        attribute: node.name,
    };
    { type: "delete relationship", relation: node.name };
    {
        type: "delete attribute",
        object: objectId,
        attribute: attributeName,
    };
    {
        type: "delete relationship",
        source: d.source.id,
        target: d.target.id,
        relation: d.relation,
    };
    {
        type: "delete relationship",
        source: d.source.id,
        target: d.target.id,
        relation: d.relation,
    };
    """
    
    # 1. Generate object id - mask mapping
    object_id_mask = {}
    print("Original SG", original_sg)
    for node in original_sg["objects"]:
        print("\tNode", node)
        object_id_mask[node["id"]] = node["name"]
            
    print("Object ID Mask", object_id_mask)
    
    # 2. Generate object id - prompt mapping
    object_id_prompt = {}
    
    for change in graph_changes:
        print("Change", change)
        if(change["type"] == "add object"):
            object_id_prompt[change["object"]] = change["name"]
        elif(change["type"] == "add attribute"):
            object_id_prompt[change["object"]] = change["attribute"] + " " + object_id_mask[change["object"]]
        elif(change["type"] == "add relationship"):
            pass
        elif(change["type"] == "rename object"):
            # find object name in object_id_mask and replace it with new name
            object_id = change["object"]
            for node in original_sg["objects"]:
                if(node["id"] == object_id):
                    original_name = object_id_mask[object_id]
            
            print("Object ID", object_id)
            print("Original Name", original_name)
            print("Object ID Mask", object_id_mask[object_id])
            object_id_prompt[object_id] = object_id_mask[object_id].replace(original_name, change["name"])
        elif(change["type"] == "rename attribute"):
            # find attribute name in object_id_mask and replace it with new name
            object_id_prompt[change["object"]] = object_id_mask[change["object"]].replace(change["attribute"], change["name"])
        elif(change["type"] == "rename relationship"):
            pass
        elif(change["type"] == "delete object"):
            object_id_prompt[change["object"]] = "fill in the background"
        elif(change["type"] == "delete attribute"):
            object_id_prompt[change["object"]] = object_id_mask[change["object"]].replace(change["attribute"], "")
        elif(change["type"] == "delete relationship"):
            pass
    
    print("Object ID Prompt", object_id_prompt)
    
    prompt_and_mask = []
    for object_id in object_id_prompt:
        prompt_and_mask.append((object_id_prompt[object_id], object_id_mask[object_id]))
        

    return prompt_and_mask