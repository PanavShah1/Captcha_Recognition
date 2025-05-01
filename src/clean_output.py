def clean_output(x):
    fin = ""
    for j in range(len(x)-1):
        if x[j] == "°":
            continue
        elif x[j] != "°":
            if x[j+1] != x[j]:
                fin = fin + x[j]
        
    return fin

if __name__ == "__main__":
    x = "°°°°°°°°°°°°°°44°°°°°°°°°°E°°°°°°°°°°h°°°°°°°°°°°Q°°°°°°°°°°°°°°°°°°°°°°°°°"
    result = clean_output(x)
    print(result)  
