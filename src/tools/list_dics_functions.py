
def add_listofdics_to_dic(dic,listofdics):
    """
        Add values from a list of single-key dictionaries to a flat dictionary.

        Each dictionary in the list corresponds to a key in `dic`. The value
        from the single-item dictionary replaces the current value in `dic`.

        PARAMETERS
        ----------
        dic : dict
            Dictionary to update.

        listofdics : list of dict
            List of single-key dictionaries. Each dictionary provides
            the new value for the corresponding key in `dic`.

        RETURNS
        -------
        None
            The dictionary is updated in place.
        """
    for base_key, item in zip(dic.keys(), listofdics):
        value = list(item.values())[0]
        dic[base_key] = value

def add_listofdics_to_dicofdics(dic,listofdics,new_keys):
    """
        Add values from a list of dictionaries to a dictionary of dictionaries,
        optionally renaming keys.

        PARAMETERS
        ----------
        dic : dict of dict
            Base dictionary where each value is a sub-dictionary.

        listofdics : list of dict
            List of dictionaries to add to each sub-dictionary.

        new_keys : list
            New key names corresponding to the old keys in listofdics.

        RETURNS
        -------
        None
            The dictionary of dictionaries is updated in place.
        """
    # Update each sub-dictionary with the corresponding dictionary from the list
    for i, j in zip(dic, listofdics):
        dic[i].update(j)

    # Collect old keys from listofdics
    keys_list = list(set(key for dictionary in listofdics for key in dictionary.keys()))

    # Rename keys in each sub-dictionary
    for i in range(len(keys_list)):
        change_keys_dic(dic, keys_list[i], new_keys[i])

def add_dic_to_dicofdics(dicofdics,dic,key):
    """
        Add values from a flat dictionary to each sub-dictionary in a
        dictionary of dictionaries under a single specified key.

        PARAMETERS
        ----------
        dicofdics : dict of dict
            Dictionary of dictionaries to update.

        dic : dict
            Flat dictionary providing values to add.

        key : hashable
            Key under which to store the value in each sub-dictionary.

        RETURNS
        -------
        None
            The dictionary of dictionaries is updated in place.
        """
    for clave in dicofdics:
            dicofdics[clave][key] = dic[clave]

def product_columns_dic(dic,key1,key2):
    """
        Multiply two fields for each sub-dictionary in a dictionary and
        return a new dictionary with the products.

        PARAMETERS
        ----------
        dic : dict of dict
            Dictionary of dictionaries containing numeric values.

        key1, key2 : hashable
            Keys corresponding to the fields to multiply.

        RETURNS
        -------
        dict
            Dictionary where each key maps to the product of the two specified fields.
        """
    product_dic = {}
    for key, sub_dic in dic.items():
        product = round(sub_dic[key1] * sub_dic[key2], 3)
        product_dic[key] = product
    return product_dic

def change_keys_dic(dic,old_key,new_key):
    """
        Rename a key in all sub-dictionaries of a dictionary.

        PARAMETERS
        ----------
        dic : dict of dict
            Dictionary of dictionaries whose keys are to be renamed.

        old_key : hashable
            Original key name.

        new_key : hashable
            New key name to replace old_key.

        RETURNS
        -------
        None
            The dictionary of dictionaries is updated in place.
        """
    for sub_dic in dic.values():
            sub_dic[new_key] = sub_dic.pop(old_key)

def add_value_to_dicofdics(dic,key,value):
    """
        Add the same key–value pair to every sub-dictionary inside a
        dictionary of dictionaries.

        PARAMETERS
        ----------
        dic : dict
            Dictionary whose values are themselves dictionaries.

        key : hashable
            Key to insert into each sub-dictionary.

        value : any
            Value assigned to the given key in all sub-dictionaries.

        RETURNS
        -------
        None
            The function modifies the input dictionary in place.
        """
    for sub_dic in dic.values():
        sub_dic[key] = value

def column_sum(dic, key):
    """
        Compute the sum of values for a specified key across all
        sub-dictionaries in a dictionary of dictionaries.

        PARAMETERS
        ----------
        dic : dict of dict
            Dictionary of dictionaries containing numeric values.

        key : hashable
            Key corresponding to the field to sum.

        RETURNS
        -------
        float
            Sum of the specified field across all sub-dictionaries.
        """
    summation = 0
    for entry in dic.values():
        if key in entry:
            summation += entry[key]
    return summation


