using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class FirefighterController : MonoBehaviour
{
    
    private CharacterController cc;
    public float moveSpeed;
    public float jumpSpeed;
    private float horizontalMove, verticalMove;
    private Vector3 dir;
    
    // Start is called before the first frame update
    void Start()
    {
        cc=GetComponent<CharacterController>();
    }

    // Update is called once per frame
    void Update()
    {
        horizontalMove=Input.GetAxis("Horizontal")*moveSpeed;
		verticalMove=Input.GetAxis("Vertical")*moveSpeed;

		dir=transform.forward*verticalMove+transform.right*horizontalMove;
		cc.Move(dir*Time.deltaTime);
    }

    private void OnCollisionEnter(Collision other)
    {
        Debug.Log(other.gameObject.name);
    }
}
