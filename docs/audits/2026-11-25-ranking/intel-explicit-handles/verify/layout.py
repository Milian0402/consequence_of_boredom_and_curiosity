"""Independent observed physical-layout checks for the unrelated grouped-B native worker."""
def require(ok,why):
    if not ok: raise ValueError(why)

def verify_layout(panel_row,record,page_size):
    """Return the observed A/B/C address residues and the number of protected C parent cells."""
    require(isinstance(page_size,int) and page_size>=64 and page_size&(page_size-1)==0,"Unsupported native page size")
    m,n,k=(int(panel_row[x]) for x in ("m","n","k"))
    require(min(m,n,k)>0,"Invalid GEMM dimensions")
    residues=[];counts=[]
    for name,row_count,col_count in [("a",m,k),("b",k,n),("c",m,n)]:
        ld=int(panel_row["ld"+name]); r=int(panel_row[name+"r"]); c=int(panel_row[name+"c"])
        require(min(r,c)>=0 and ld>=c+col_count and int(record["ld"+name])==ld,"Wrong parent leading dimension "+name)
        residue=4*(r*ld+c)%64
        require(int(record[name+"_mod64"])==residue,"Observed data-pointer residue differs from declared storage "+name)
        total_bytes=4*(r+row_count+2)*ld
        allocated=((total_bytes+page_size-1)//page_size)*page_size//4
        require((r+row_count-1)*ld+c+col_count<=allocated,"Parent rectangle does not fit")
        residues.append(residue);counts.append(allocated)
    untouched=counts[2]-m*n
    require(int(record["checked"])==m*n and int(record["guards"])==untouched,"Incomplete C output/parent guard coverage")
    return residues,untouched
