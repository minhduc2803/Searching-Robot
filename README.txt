Có 4 file có thể run là astar.py, dijkstra.py, bestfirstsearch.py và level3.py.
*Các file run buộc phải ở chung thư mục với file graph.py
Chỉ cần nhấn run là chạy được thuật toán.
Mỗi file chạy một thuật toán, thể hiện kết quả ở console và thể hiện đồ họa bằng thư viện matplotlib
Vì vậy buộc phải cài thư viện matplotlib trước.

Các file input nhóm em tự thiết kế, nếu muốn thay đổi input cho thuật toán thì đổi tên file
tại dòng gọi hàm init_world('file_name').
*File input buộc phải đặt chung thư mục với file run.

Các file input1.txt, input2.txt, input3.txt và input4.txt là dành cho mức 1, mức 2
file input31.txt input32.txt input33.txt là dành cho mức 3

Thời gian thể hiện đồ họa không phải là thời gian thực hiện thuật toán (thời gian thực hiện thuật toán
nằm ở console).
Nếu thầy muốn tăng nhanh quá trình hiển thị đồ họa thì có thể thay đổi biến n_show trước khi gọi hàm
display_process(). Ví dụ n_show = n_show*2 thì phần đồ họa sẽ hiển thị nhanh gấp đôi, còn muốn chậm
lại có thể thêm n_show = int(n_show/2). Biến n_show nhóm em đã tính toán để thể hiện được tương quan 
thời gian chạy giữa các thuật toán.