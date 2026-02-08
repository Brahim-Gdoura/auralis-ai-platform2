import re
import json

# 🎨 VOS 3 IMAGES PAR DÉFAUT (une pour chaque produit)
PRODUCT_IMAGES = [
    "https://www.oliveandcocoa.com/images/uploads/17592_Pink_Cashmere_Scarf_P.jpg",
    "https://i.ebayimg.com/images/g/gV8AAOSwgLdj18J2/s-l400.jpg",
    "https://m.media-amazon.com/images/I/81RYl3u2xLL._AC_UY1000_.jpg",
]


def get_product_image(index):
    """
    Retourne une image différente selon l'index du produit.
    """
    image_index = index % len(PRODUCT_IMAGES)
    return PRODUCT_IMAGES[image_index]


def is_product_relevant(product, query, response):
    """
    Détermine si un produit est pertinent pour la requête de l'utilisateur.
    
    Args:
        product: Dictionnaire contenant les infos du produit
        query: Question de l'utilisateur
        response: Réponse du chatbot
        
    Returns:
        bool: True si le produit est pertinent, False sinon
    """
    query_lower = query.lower()
    response_lower = response.lower()
    
    product_name = product.get('name', '').lower()
    product_category = product.get('category', '').lower()
    product_description = product.get('description', '').lower()
    
    # Combiner tous les textes du produit
    product_text = f"{product_name} {product_category} {product_description}"
    
    # 1. Vérifier si le nom du produit est mentionné dans la requête ou la réponse
    if product_name and len(product_name) > 3:
        # Chercher des mots du nom du produit
        product_words = product_name.split()
        for word in product_words:
            if len(word) > 3 and (word in query_lower or word in response_lower):
                print(f"  ✓ Produit pertinent: '{product_name}' - mot '{word}' trouvé")
                return True
    
    # 2. Vérifier si la catégorie est mentionnée
    if product_category and len(product_category) > 3:
        if product_category in query_lower or product_category in response_lower:
            print(f"  ✓ Produit pertinent: '{product_name}' - catégorie '{product_category}' trouvée")
            return True
    
    # 3. Chercher des mots-clés communs (au moins 2 mots en commun)
    query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
    response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
    product_words = set(re.findall(r'\b\w{4,}\b', product_text))
    
    # Mots à ignorer (stop words)
    stop_words = {'this', 'that', 'with', 'have', 'from', 'they', 'were', 'been', 
                  'have', 'your', 'more', 'will', 'other', 'about', 'which', 'their',
                  'would', 'these', 'there', 'could', 'than', 'then', 'some', 'what'}
    
    query_words -= stop_words
    response_words -= stop_words
    product_words -= stop_words
    
    # Mots communs avec la requête
    common_with_query = query_words & product_words
    common_with_response = response_words & product_words
    
    if len(common_with_query) >= 1 or len(common_with_response) >= 2:
        print(f"  ✓ Produit pertinent: '{product_name}' - mots communs: {common_with_query | common_with_response}")
        return True
    
    # 4. Recherche de synonymes et mots-clés spécifiques
    keyword_mappings = {
        'gift': ['present', 'birthday', 'anniversary', 'christmas', 'mother', 'father', 'mom', 'dad'],
        'tech': ['electronic', 'technology', 'gadget', 'device', 'phone', 'computer', 'laptop'],
        'clothes': ['clothing', 'shirt', 'pants', 'dress', 'fashion', 'wear'],
        'home': ['house', 'kitchen', 'furniture', 'decor', 'decoration'],
        'sport': ['fitness', 'exercise', 'workout', 'gym', 'yoga', 'running'],
    }
    
    for key, synonyms in keyword_mappings.items():
        if key in product_text:
            for synonym in synonyms:
                if synonym in query_lower or synonym in response_lower:
                    print(f"  ✓ Produit pertinent: '{product_name}' - synonyme '{synonym}' pour '{key}'")
                    return True
    
    print(f"  ✗ Produit NON pertinent: '{product_name}' - aucune correspondance trouvée")
    return False


def extract_products_from_response(response_text, source_documents, user_query=""):
    """
    Extrait les informations de produits depuis la réponse et les documents sources.
    Filtre pour ne retourner que les produits pertinents à la requête.
    
    Args:
        response_text: Texte de la réponse du chatbot
        source_documents: Documents sources du RAG
        user_query: Question de l'utilisateur (pour le filtrage)
        
    Returns:
        Liste de produits pertinents et filtrés
    """
    all_products = []
    
    print(f"\n🔍 Extraction des produits pour la requête: '{user_query}'")
    
    # Chercher dans les documents sources pour les informations de produits
    for doc in source_documents:
        content = doc.page_content
        
        # Parser les informations de produit depuis le contenu
        product_blocks = content.split("---")
        
        for block in product_blocks:
            if "Product:" in block:
                product = {}
                
                # Extraire le nom du produit
                name_match = re.search(r'Product:\s*(.+?)(?:\n|$)', block)
                if name_match:
                    product['name'] = name_match.group(1).strip()
                
                # Extraire la catégorie
                category_match = re.search(r'Category:\s*(.+?)(?:\n|$)', block)
                if category_match:
                    product['category'] = category_match.group(1).strip()
                
                # Extraire le prix
                price_match = re.search(r'Price:\s*\$?([0-9.]+)', block)
                if price_match:
                    product['price'] = price_match.group(1)
                
                # Extraire le stock
                stock_match = re.search(r'Stock:\s*([0-9]+)', block)
                if stock_match:
                    product['stock'] = int(stock_match.group(1))
                
                # Extraire la description
                desc_match = re.search(r'Description:\s*(.+?)(?:\n|$)', block)
                if desc_match:
                    product['description'] = desc_match.group(1).strip()
                
                # Extraire l'ID du produit
                id_match = re.search(r'ID:\s*(.+?)(?:\n|$)', block)
                if id_match:
                    product['product_id'] = id_match.group(1).strip()
                
                # Ajouter un badge si le stock est faible
                if product.get('stock', 0) < 10 and product.get('stock', 0) > 0:
                    product['badge'] = "Low Stock"
                elif product.get('stock', 0) == 0:
                    product['badge'] = "Out of Stock"
                
                # Ajouter une note par défaut
                product['rating'] = "4.5"
                
                if product.get('name'):
                    all_products.append(product)
    
    print(f"📦 Total de produits trouvés dans les documents: {len(all_products)}")
    
    # 🎯 FILTRAGE INTELLIGENT : Ne garder que les produits pertinents
    relevant_products = []
    
    if user_query:
        print("\n🎯 Filtrage des produits pertinents:")
        for product in all_products:
            if is_product_relevant(product, user_query, response_text):
                relevant_products.append(product)
    else:
        # Si pas de requête fournie, garder tous les produits (comportement par défaut)
        relevant_products = all_products
    
    print(f"\n✅ Produits pertinents retenus: {len(relevant_products)}/{len(all_products)}")
    
    # Dédupliquer les produits par nom
    unique_products = []
    seen_names = set()
    for product in relevant_products:
        if product['name'] not in seen_names:
            unique_products.append(product)
            seen_names.add(product['name'])
    
    # Limiter à 3 produits max
    unique_products = unique_products[:3]
    
    # 🎨 ASSIGNER UNE IMAGE DIFFÉRENTE À CHAQUE PRODUIT
    for index, product in enumerate(unique_products):
        product['image'] = get_product_image(index)
        print(f"  → Produit {index + 1}: {product['name']} [Image {index + 1}]")
    
    return unique_products


def should_show_products(query, response):
    """
    Détermine si on devrait afficher des cartes de produits
    """
    # Mots-clés qui indiquent une recherche de produit
    product_keywords = [
        'product', 'item', 'buy', 'purchase', 'recommend', 'suggest',
        'looking for', 'need', 'want', 'show me', 'find', 'gift',
        'price', 'cost', 'available', 'stock', 'sell'
    ]
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Vérifier si la requête ou la réponse contient des mots-clés de produit
    has_product_keywords = any(keyword in query_lower or keyword in response_lower 
                               for keyword in product_keywords)
    
    # Vérifier si la réponse mentionne un prix
    has_price = '$' in response or 'price' in response_lower
    
    return has_product_keywords or has_price