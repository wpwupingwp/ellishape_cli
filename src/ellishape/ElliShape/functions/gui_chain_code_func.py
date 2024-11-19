import numpy as np

def gui_chain_code_func(axis_info, oringin_ori):
    nrow, ncol = axis_info.shape
    # print(nrow)
    # print(ncol)
    idxall = np.where(axis_info == 255)
    print(idxall)
    # idxall=np.transpose(idxall)
    # print(idxall)
    numoftotalpoints = len(idxall[0])

    rowidxst = oringin_ori[0]
    colidxst = oringin_ori[1]
    oringin_ori = [rowidxst, colidxst]
    
    flags = np.zeros((nrow, ncol))
    contour_points = np.zeros((numoftotalpoints, 2))
    # print(contour_points)
    backword_points = np.zeros((numoftotalpoints, 2))
    chain_code_ori = np.zeros(numoftotalpoints, dtype=int)
    
    numofpoints = 0
    numofpoints_pre = 0
    contour_points[numofpoints] = oringin_ori
    flags[rowidxst, colidxst] = 1
    numofbackword = 0
    backwordflag = False
    
    while numofpoints < numoftotalpoints-1:
        fatecount = 8
        # print(numofpoints)
        rowidxpre, colidxpre = contour_points[numofpoints]#前一个点的坐标
        
        rowidx = int(min(max(rowidxpre, 0), nrow - 1)) #
        colidx = int(min(max(colidxpre + 1,  0), ncol-1))
        # print(rowidx)
        # print(colidx)
        # print(rowidxst)
        # print(colidxst)
        
        if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
            numofpoints += 1
            contour_points[numofpoints] = [rowidx, colidx]
            chain_code_ori[numofpoints - 1] = 0
            break

        # print(axis_info[rowidx, colidx])
        # print(flags[rowidx, colidx])
        if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
            numofpoints += 1
            # print(numofpoints)
            contour_points[numofpoints] = [rowidx, colidx]
            flags[rowidx, colidx] = 1
            chain_code_ori[numofpoints - 1] = 0
        else:
            fatecount -= 1
            rowidx = int(min(max(rowidxpre - 1, 0), nrow-1))
            colidx = int(min(max(colidxpre + 1, 0), ncol-1))
            if rowidx == oringin_ori[0] and colidx == oringin_ori[1] and numofpoints > 1:
                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                chain_code_ori[numofpoints - 1] = 1
                break
            # print(rowidx)
            # print(colidx)
            # print(axis_info[colidx,rowidx])
            # print(flags[rowidx, colidx])
            if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:

                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                flags[rowidx, colidx] = 1
                chain_code_ori[numofpoints - 1] = 1
            else:
                fatecount -= 1
                rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                colidx = int(min(max(colidxpre, 0), ncol - 1))
                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    chain_code_ori[numofpoints - 1] = 2
                    break
                if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    flags[rowidx, colidx] = 1
                    chain_code_ori[numofpoints - 1] = 2
                else:
                    fatecount -= 1
                    rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                    colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        chain_code_ori[numofpoints - 1] = 3
                        break
                    if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        flags[rowidx, colidx] = 1
                        chain_code_ori[numofpoints - 1] = 3
                    else:
                        fatecount -= 1
                        rowidx = int(min(max(rowidxpre, 0), nrow - 1))
                        colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                        if rowidx == rowidxst and colidx == colidxst and numofpoints > 2:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            chain_code_ori[numofpoints - 1] = 4
                            break
                        if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            flags[rowidx, colidx] = 1
                            chain_code_ori[numofpoints - 1] = 4
                        else:
                            fatecount -= 1
                            rowidx = int(min(max(rowidxpre + 1, 0), nrow - 1))
                            colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                            if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                chain_code_ori[numofpoints - 1] = 5
                                break
                            if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                flags[rowidx, colidx] = 1
                                chain_code_ori[numofpoints - 1] = 5
                            else:
                                fatecount -= 1
                                rowidx = int(min(max(rowidxpre + 1, 0), nrow - 1))
                                colidx = int(min(max(colidxpre, 0), ncol - 1))
                                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx, colidx]
                                    chain_code_ori[numofpoints - 1] = 6
                                    break
                                if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx, colidx]
                                    flags[rowidx, colidx] = 1
                                    chain_code_ori[numofpoints - 1] = 6
                                else:
                                    fatecount -= 1
                                    rowidx = int(min(max(rowidxpre + 1, 0), nrow - 1))
                                    colidx = int(min(max(colidxpre + 1, 0), ncol - 1))
                                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx, colidx]
                                        chain_code_ori[numofpoints - 1] = 7
                                        break
                                    if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx, colidx]
                                        flags[rowidx, colidx] = 1
                                        chain_code_ori[numofpoints - 1] = 7
                                    else:
                                        fatecount -= 1
        
        if fatecount == 0:
            rowidxpre = int(contour_points[numofpoints, 0])
            colidxpre = int(contour_points[numofpoints, 1])
            # print(rowidxpre)
            # print(colidxpre)
            axis_info[rowidxpre, colidxpre] = 0
            if numofbackword == 0 and numofpoints > numofpoints_pre:
                numofpoints_pre = numofpoints
                backwordflag = 1
            numofpoints -= 1
            if numofpoints < 1:
                break
            numofbackword += 1
            if numofbackword > 20:
                break
            else:
                if backwordflag == 1:
                    backword_points[numofbackword] = [rowidxpre, colidxpre]
        else:
            numofbackword = 0
            backwordflag = False
    
    chain_code = chain_code_ori[:numofpoints]
    print(chain_code)
    # endpoint = backword_points[0] 
    oringin = oringin_ori
    print(oringin)
    # print(endpoint)
    # print((np.where(flags == 1)))
    return chain_code, oringin
