def write_data_to_file(data, filename):

    # Open a text file to write data
    with open(filename, 'w') as file:
        if data.ndim>1:
            # Writes data to a text file by line
            for row in data:
                # print(row)
                file.write(' '.join(map(str, row)) + '\n')
        else:
            file.write(' '.join(map(str, data)) + '\n')
    file.close()