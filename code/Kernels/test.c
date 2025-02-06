

#include <stdio.h>

int main() {
    FILE *out = fopen("out.csv", "w"); 
    fprintf(out, "Hello World.\n");
    fclose(out);
}